/*
  Project: Object Classifier by Color
  Board:   Arduino Nano 33 BLE Sense
  Sensor:  APDS-9960 (onboard)
  Task:    Classify Apple, Banana, Orange from RGB using TensorFlow Lite Micro
  Notes:
    - Input features: ["Red", "Green", "Blue"] normalized to [0,1], order: R,G,B
    - Moving average smoothing window: 4
    - Inference rate: ~10 Hz (every 100 ms)
    - Outputs predicted class and emoji over Serial (9600 baud)
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro core and components
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// The model binary as a C array
#include "model.h"

// ---------- Configuration ----------
static const int kTensorArenaSize = 16384;
static const uint32_t kBaudRate = 9600;
static const uint8_t kNumChannels = 3;            // Red, Green, Blue
static const uint8_t kNumClasses  = 3;            // Apple, Banana, Orange
static const uint8_t kSmoothingWindow = 4;        // moving average window
static const uint32_t kInferencePeriodMs = 100;   // 10 Hz

// ---------- Globals (TFLM) ----------
tflite::ErrorReporter* g_error_reporter = nullptr;
const tflite::Model* g_tfl_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

// Provide a statically allocated tensor arena for TFLM
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

// ---------- Globals (App) ----------
static const char* kClassNames[kNumClasses] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};

static float g_buffer_r[kSmoothingWindow] = {0};
static float g_buffer_g[kSmoothingWindow] = {0};
static float g_buffer_b[kSmoothingWindow] = {0};
static float g_sum_r = 0.0f, g_sum_g = 0.0f, g_sum_b = 0.0f;
static uint8_t g_buf_index = 0;
static uint8_t g_buf_count = 0;

static uint32_t g_next_inference_ms = 0;

// ---------- Helpers ----------
static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static inline int clamp255(int x) {
  if (x < 0) return 0;
  if (x > 255) return 255;
  return x;
}

static void smoothingPush(float r, float g, float b, float& out_r, float& out_g, float& out_b) {
  // Remove old
  g_sum_r -= g_buffer_r[g_buf_index];
  g_sum_g -= g_buffer_g[g_buf_index];
  g_sum_b -= g_buffer_b[g_buf_index];

  // Insert new
  g_buffer_r[g_buf_index] = r;
  g_buffer_g[g_buf_index] = g;
  g_buffer_b[g_buf_index] = b;

  g_sum_r += r;
  g_sum_g += g;
  g_sum_b += b;

  if (g_buf_count < kSmoothingWindow) g_buf_count++;

  g_buf_index++;
  if (g_buf_index >= kSmoothingWindow) g_buf_index = 0;

  float denom = static_cast<float>(g_buf_count);
  out_r = g_sum_r / denom;
  out_g = g_sum_g / denom;
  out_b = g_sum_b / denom;
}

static void fillInputTensor(float r, float g, float b) {
  // Supports float32 and quantized (uint8 / int8) input tensors
  if (!g_input) return;

  if (g_input->type == kTfLiteFloat32) {
    // Expected shape [1, 3]; write in order: R, G, B
    g_input->data.f[0] = r;
    g_input->data.f[1] = g;
    g_input->data.f[2] = b;
  } else if (g_input->type == kTfLiteUInt8) {
    // Quantize using scale/zero_point if provided; otherwise map [0,1] -> [0,255]
    const float scale = g_input->params.scale == 0 ? (1.0f / 255.0f) : g_input->params.scale;
    const int32_t zp = g_input->params.zero_point;
    auto quantize = [&](float v) -> uint8_t {
      int32_t q = static_cast<int32_t>(roundf(v / scale) + zp);
      if (q < 0) q = 0;
      if (q > 255) q = 255;
      return static_cast<uint8_t>(q);
    };
    g_input->data.uint8[0] = quantize(r);
    g_input->data.uint8[1] = quantize(g);
    g_input->data.uint8[2] = quantize(b);
  } else if (g_input->type == kTfLiteInt8) {
    const float scale = g_input->params.scale == 0 ? (1.0f / 127.0f) : g_input->params.scale;
    const int32_t zp = g_input->params.zero_point; // typically -128..127
    auto quantize = [&](float v) -> int8_t {
      int32_t q = static_cast<int32_t>(roundf(v / scale) + zp);
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      return static_cast<int8_t>(q);
    };
    g_input->data.int8[0] = quantize(r);
    g_input->data.int8[1] = quantize(g);
    g_input->data.int8[2] = quantize(b);
  } else {
    // Unsupported input type; do nothing
  }
}

static int argmaxAndConfidence(float& confidence_out) {
  // Reads g_output and returns index of max score.
  // Tries to compute confidence using output scale/zero_point when quantized, or raw float if float32.
  confidence_out = 0.0f;
  if (!g_output) return -1;

  int best_idx = 0;

  if (g_output->type == kTfLiteFloat32) {
    float best_val = g_output->data.f[0];
    for (int i = 1; i < kNumClasses; i++) {
      float v = g_output->data.f[i];
      if (v > best_val) {
        best_val = v;
        best_idx = i;
      }
    }
    confidence_out = best_val; // assume probabilities or scores in [0,1]
  } else if (g_output->type == kTfLiteUInt8) {
    uint8_t best_val = g_output->data.uint8[0];
    for (int i = 1; i < kNumClasses; i++) {
      uint8_t v = g_output->data.uint8[i];
      if (v > best_val) {
        best_val = v;
        best_idx = i;
      }
    }
    // Dequantize best value if scale available
    float scale = g_output->params.scale;
    int32_t zp = g_output->params.zero_point;
    if (scale > 0.0f) {
      confidence_out = scale * (static_cast<int32_t>(g_output->data.uint8[best_idx]) - zp);
    } else {
      confidence_out = best_val / 255.0f;
    }
  } else if (g_output->type == kTfLiteInt8) {
    int8_t best_val = g_output->data.int8[0];
    for (int i = 1; i < kNumClasses; i++) {
      int8_t v = g_output->data.int8[i];
      if (v > best_val) {
        best_val = v;
        best_idx = i;
      }
    }
    float scale = g_output->params.scale;
    int32_t zp = g_output->params.zero_point;
    if (scale > 0.0f) {
      confidence_out = scale * (static_cast<int32_t>(g_output->data.int8[best_idx]) - zp);
    } else {
      // Map int8 [-128,127] roughly to [0,1] for display
      confidence_out = (static_cast<int>(g_output->data.int8[best_idx]) + 128) / 255.0f;
    }
  } else {
    // Unknown type; just return index 0
    best_idx = 0;
    confidence_out = 0.0f;
  }

  // Clamp confidence to [0,1] for printing
  if (confidence_out < 0.0f) confidence_out = 0.0f;
  if (confidence_out > 1.0f) confidence_out = 1.0f;

  return best_idx;
}

void setup() {
  Serial.begin(kBaudRate);
  while (!Serial && millis() < 4000) { /* wait for serial up to 4s */ }

  // Initialize sensor
  if (!APDS.begin()) {
    Serial.println("APDS-9960 init failed. Check wiring or board.");
    while (true) { delay(1000); }
  }
  Serial.println("APDS-9960 ready.");

  // Setup TFLM error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  g_error_reporter = &micro_error_reporter;

  // Map the model from the binary array
  g_tfl_model = tflite::GetModel(model); // 'model' comes from model.h
  if (g_tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch. Model version: ");
    Serial.print(g_tfl_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Create interpreter (placed statically so it persists)
  static tflite::MicroInterpreter static_interpreter(
      g_tfl_model, resolver, g_tensor_arena, kTensorArenaSize, g_error_reporter);
  g_interpreter = &static_interpreter;

  // Allocate TFLM tensors
  TfLiteStatus alloc_status = g_interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // Cache input/output tensor pointers
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  // Basic input shape/type check (informative)
  if (g_input) {
    Serial.print("Input dims: ");
    for (int i = 0; i < g_input->dims->size; i++) {
      Serial.print(g_input->dims->data[i]);
      if (i < g_input->dims->size - 1) Serial.print("x");
    }
    Serial.print(" type=");
    Serial.println(g_input->type);
  }
  if (g_output) {
    Serial.print("Output dims: ");
    for (int i = 0; i < g_output->dims->size; i++) {
      Serial.print(g_output->dims->data[i]);
      if (i < g_output->dims->size - 1) Serial.print("x");
    }
    Serial.print(" type=");
    Serial.println(g_output->type);
  }

  // Initialize smoothing buffers
  for (uint8_t i = 0; i < kSmoothingWindow; i++) {
    g_buffer_r[i] = g_buffer_g[i] = g_buffer_b[i] = 0.0f;
  }
  g_sum_r = g_sum_g = g_sum_b = 0.0f;
  g_buf_index = 0;
  g_buf_count = 0;

  g_next_inference_ms = millis();

  Serial.println("TinyML color classifier ready.");
}

void loop() {
  // Run ~10 Hz
  const uint32_t now = millis();
  if (now < g_next_inference_ms) {
    delay(1);
    return;
  }
  g_next_inference_ms = now + kInferencePeriodMs;

  // Make sure color data is available
  if (!APDS.colorAvailable()) {
    return; // try next cycle
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  APDS.readColor(r_raw, g_raw, b_raw);

  // Clamp to sensor's nominal range and normalize to [0,1]
  r_raw = clamp255(r_raw);
  g_raw = clamp255(g_raw);
  b_raw = clamp255(b_raw);

  float r = clamp01(static_cast<float>(r_raw) / 255.0f);
  float g = clamp01(static_cast<float>(g_raw) / 255.0f);
  float b = clamp01(static_cast<float>(b_raw) / 255.0f);

  // Moving average smoothing
  float r_s, g_s, b_s;
  smoothingPush(r, g, b, r_s, g_s, b_s);

  // Feed model input
  fillInputTensor(r_s, g_s, b_s);

  // Inference
  if (g_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Decode result
  float confidence = 0.0f;
  int idx = argmaxAndConfidence(confidence);
  if (idx < 0 || idx >= kNumClasses) {
    Serial.println("Invalid prediction index");
    return;
  }

  // Print prediction with emoji
  Serial.print("Pred: ");
  Serial.print(kClassNames[idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[idx]);
  Serial.print("  conf=");
  Serial.print(confidence * 100.0f, 1);
  Serial.print("%  RGB(norm, smoothed)=(");
  Serial.print(r_s, 3); Serial.print(", ");
  Serial.print(g_s, 3); Serial.print(", ");
  Serial.print(b_s, 3); Serial.println(")");
}