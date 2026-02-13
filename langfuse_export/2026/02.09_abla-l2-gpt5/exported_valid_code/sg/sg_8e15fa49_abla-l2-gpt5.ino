/*
  Project: Color Object Classifier (Emoji over Serial)
  Board:   Arduino Nano 33 BLE Sense
  Sensor:  APDS9960 (onboard) - RGB channels used
  Model:   TensorFlow Lite Micro (included via model.h)

  Notes:
  - Input expected as normalized RGB in [0.0, 1.0] with R+G+B ≈ 1.0 (matches dataset).
  - Outputs class scores for ["Apple", "Banana", "Orange"].
  - Prints predicted label and emoji over Serial.
*/

#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro headers (Arduino_TensorFlowLite library)
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Compiled TFLM model bytes
#include "model.h"  // Provides: const unsigned char model[];  (do not redeclare)

// ======== Application constants ========
static const uint32_t kBaudRate          = 9600;
static const uint32_t kReadIntervalMs    = 200;
static const int      kWarmupReadings    = 5;
static const int      kNumClasses        = 3;
static const char*    kLabels[kNumClasses] = {"Apple", "Banana", "Orange"};
static const char*    kEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};

// ======== TFLM globals ========
constexpr int kTensorArenaSize = 16384;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

tflite::ErrorReporter*        g_error_reporter = nullptr;
const tflite::Model*          g_model          = nullptr;
tflite::MicroInterpreter*     g_interpreter    = nullptr;
TfLiteTensor*                 g_input          = nullptr;
TfLiteTensor*                 g_output         = nullptr;

// Keep these static so their lifetimes outlive setup()
static tflite::MicroErrorReporter g_micro_error_reporter;
static tflite::AllOpsResolver     g_resolver;  // Registers all ops to avoid missing operator issues

// ======== Runtime control ========
static uint32_t g_last_read_ms = 0;
static int      g_warmup_left  = kWarmupReadings;

// ======== Helpers ========
static int argmax(const float* vals, int n) {
  int idx = 0;
  float best = vals[0];
  for (int i = 1; i < n; ++i) {
    if (vals[i] > best) { best = vals[i]; idx = i; }
  }
  return idx;
}

static void fillInputTensor(float r, float g, float b) {
  // Handle float32 and quantized models
  switch (g_input->type) {
    case kTfLiteFloat32: {
      // Expected shape [1, 3]
      float* data = g_input->data.f;
      data[0] = r;
      data[1] = g;
      data[2] = b;
      break;
    }
    case kTfLiteUInt8: {
      const float  scale = g_input->params.scale;
      const int32_t zp   = g_input->params.zero_point;
      uint8_t* data = g_input->data.uint8;
      auto q = [&](float x) -> uint8_t {
        int32_t qv = static_cast<int32_t>(roundf(x / scale) + zp);
        if (qv < 0) qv = 0;
        if (qv > 255) qv = 255;
        return static_cast<uint8_t>(qv);
      };
      data[0] = q(r);
      data[1] = q(g);
      data[2] = q(b);
      break;
    }
    case kTfLiteInt8: {
      const float  scale = g_input->params.scale;
      const int32_t zp   = g_input->params.zero_point;
      int8_t* data = g_input->data.int8;
      auto q = [&](float x) -> int8_t {
        int32_t qv = static_cast<int32_t>(roundf(x / scale) + zp);
        if (qv < -128) qv = -128;
        if (qv > 127)  qv = 127;
        return static_cast<int8_t>(qv);
      };
      data[0] = q(r);
      data[1] = q(g);
      data[2] = q(b);
      break;
    }
    default:
      // Unsupported type; do nothing
      break;
  }
}

static void readOutputScores(float* out_probs /* size >= kNumClasses */) {
  switch (g_output->type) {
    case kTfLiteFloat32: {
      float* data = g_output->data.f;
      for (int i = 0; i < kNumClasses; ++i) out_probs[i] = data[i];
      break;
    }
    case kTfLiteUInt8: {
      const float  scale = g_output->params.scale;
      const int32_t zp   = g_output->params.zero_point;
      uint8_t* data = g_output->data.uint8;
      for (int i = 0; i < kNumClasses; ++i) {
        // Dequantize to float
        out_probs[i] = (static_cast<int32_t>(data[i]) - zp) * scale;
      }
      break;
    }
    case kTfLiteInt8: {
      const float  scale = g_output->params.scale;
      const int32_t zp   = g_output->params.zero_point;
      int8_t* data = g_output->data.int8;
      for (int i = 0; i < kNumClasses; ++i) {
        out_probs[i] = (static_cast<int32_t>(data[i]) - zp) * scale;
      }
      break;
    }
    default: {
      for (int i = 0; i < kNumClasses; ++i) out_probs[i] = 0.0f;
      break;
    }
  }
}

// ======== Arduino lifecycle ========
void setup() {
  Serial.begin(kBaudRate);
  while (!Serial && millis() < 4000) { /* wait for serial */ }

  // Sensor init
  if (!APDS.begin()) {
    Serial.println("APDS9960 init failed. Check wiring or board.");
    while (1) { delay(100); }
  }
  // Optional: enable color mode explicitly (most builds have it by default)
  // APDS.setGestureSensitivity(80); // not needed for color
  Serial.println("APDS9960 ready.");

  // TFLM init
  g_error_reporter = &g_micro_error_reporter;

  // Load model (from model.h). Do NOT redeclare 'model' to avoid duplicate symbol error.
  g_model = tflite::GetModel(model);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema v");
    Serial.print(g_model->version());
    Serial.print(" != TFLite Micro schema v");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(100); }
  }

  // Create interpreter (keep resolver and arena alive)
  static tflite::MicroInterpreter static_interpreter(
    g_model, g_resolver, tensor_arena, kTensorArenaSize, g_error_reporter
  );
  g_interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = g_interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1) { delay(100); }
  }

  // Cache input/output tensor pointers
  g_input  = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  Serial.print("Input type: ");
  Serial.println(g_input->type);
  Serial.print("Output type: ");
  Serial.println(g_output->type);
  Serial.println("Setup complete. Warming up sensor...");
}

void loop() {
  const uint32_t now = millis();
  if ((now - g_last_read_ms) < kReadIntervalMs) {
    delay(5);
    return;
  }
  g_last_read_ms = now;

  // Wait for new color reading
  if (!APDS.colorAvailable()) {
    // No new data; try again next loop
    return;
  }

  int r_raw, g_raw, b_raw;
  APDS.readColor(r_raw, g_raw, b_raw);

  // Normalize to [0,1] with R+G+B ≈ 1.0 as in dataset
  float r = static_cast<float>(r_raw);
  float g = static_cast<float>(g_raw);
  float b = static_cast<float>(b_raw);
  float s = r + g + b;
  float rn = 0.0f, gn = 0.0f, bn = 0.0f;
  if (s > 0.0f) {
    rn = r / s;
    gn = g / s;
    bn = b / s;
  }

  // Warmup readings to stabilize sensor
  if (g_warmup_left > 0) {
    g_warmup_left--;
    if (g_warmup_left == 0) {
      Serial.println("Warmup complete. Starting inference.");
    }
    return;
  }

  // Copy features into input tensor
  fillInputTensor(rn, gn, bn);

  // Inference
  if (g_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Read output and decide
  float probs[kNumClasses];
  readOutputScores(probs);

  int best_idx = argmax(probs, kNumClasses);
  float best_p = probs[best_idx];
  const char* label = kLabels[best_idx];
  const char* emoji = kEmojis[best_idx];

  // Emit result over Serial (UTF-8 emojis supported by many terminals/Serial Monitors)
  Serial.print("RGBn: ");
  Serial.print(rn, 3); Serial.print(", ");
  Serial.print(gn, 3); Serial.print(", ");
  Serial.print(bn, 3);
  Serial.print(" | Pred: ");
  Serial.print(label);
  Serial.print(" ");
  Serial.print(emoji);
  Serial.print(" | conf: ");
  // Clamp and format confidence as percentage when it looks like a probability
  float pct = best_p;
  if (pct < 0.0f) pct = 0.0f;
  if (pct > 1.0f) pct = 1.0f;
  Serial.print(pct * 100.0f, 1);
  Serial.println("%");
}