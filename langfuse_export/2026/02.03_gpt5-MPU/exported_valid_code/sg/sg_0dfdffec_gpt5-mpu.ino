#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro - Base must come before dependent headers
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Model flatbuffer
#include "model.h"

// Application constants
static const unsigned long kBaudRate = 9600;
static const uint32_t kInferenceIntervalMs = 200;  // sampling period

// Classes and Emojis
static const char* kClassNames[3] = { "Apple", "Banana", "Orange" };
// UTF-8 emojis
static const char* EMOJI_APPLE  = "\xF0\x9F\x8D\x8E"; // 🍎
static const char* EMOJI_BANANA = "\xF0\x9F\x8D\x8C"; // 🍌
static const char* EMOJI_ORANGE = "\xF0\x9F\x8D\x8A"; // 🍊
static const char* kClassEmojis[3] = { EMOJI_APPLE, EMOJI_BANANA, EMOJI_ORANGE };

// TFLite Micro globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena
  constexpr int kTensorArenaSize = 8192;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// Utility: clamp float to [0,1]
static inline float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

// Utility: Argmax for uint8 output
static int argmax_uint8(const TfLiteTensor* out, float* conf_out) {
  int best_idx = 0;
  uint8_t best_val = 0;
  for (int i = 0; i < out->dims->data[out->dims->size - 1]; ++i) {
    uint8_t v = out->data.uint8[i];
    if (i == 0 || v > best_val) {
      best_val = v;
      best_idx = i;
    }
  }
  if (conf_out) {
    // Approximate confidence [0,1]
    *conf_out = best_val / 255.0f;
  }
  return best_idx;
}

// Utility: Argmax for float output
static int argmax_float(const TfLiteTensor* out, float* conf_out) {
  int best_idx = 0;
  float best_val = 0.0f;
  for (int i = 0; i < out->dims->data[out->dims->size - 1]; ++i) {
    float v = out->data.f[i];
    if (i == 0 || v > best_val) {
      best_val = v;
      best_idx = i;
    }
  }
  if (conf_out) {
    *conf_out = best_val;
  }
  return best_idx;
}

void setup() {
  Serial.begin(kBaudRate);
  while (!Serial) { /* wait for Serial */ }

  Serial.println("Object Classifier by Color (RGB -> NN -> Emoji)");
  Serial.println("Initializing APDS9960 color sensor...");

  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960. Check wiring and power.");
  } else {
    Serial.println("APDS9960 initialized.");
  }

  // Initialize TFLite Micro
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model from included header
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version ");
    Serial.print(tfl_model->version());
    Serial.print(" not equal to supported version ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor
  if (input->type != kTfLiteFloat32) {
    Serial.println("ERROR: Model input tensor is not float32 as expected.");
    while (1) { delay(1000); }
  }
  if (!(input->dims->size == 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.print("WARNING: Unexpected input shape: dims=");
    Serial.print(input->dims->size);
    Serial.print(" [");
    for (int i = 0; i < input->dims->size; ++i) {
      Serial.print(input->dims->data[i]);
      if (i < input->dims->size - 1) Serial.print(", ");
    }
    Serial.println("]");
  }

  // Validate output tensor (will handle uint8 or float)
  if (!(output->type == kTfLiteUInt8 || output->type == kTfLiteFloat32)) {
    Serial.println("ERROR: Model output tensor must be uint8 or float32.");
    while (1) { delay(1000); }
  }
  Serial.println("TFLite Micro initialized. Starting inference...");
  Serial.println("Columns: Red_norm, Green_norm, Blue_norm -> Prediction");
}

void loop() {
  static uint32_t last_ms = 0;
  uint32_t now = millis();
  if (now - last_ms < kInferenceIntervalMs) {
    delay(5);
    return;
  }
  last_ms = now;

  // Wait for color data
  if (!APDS.colorAvailable()) {
    // If not available yet, try again next loop
    return;
  }

  int r = 0, g = 0, b = 0, c = 0;
  // Read RGBC values
  APDS.readColor(r, g, b, c);

  // Normalize to fractions matching dataset style (~sum to 1.0)
  // Prefer c (clear/ambient) if available; fallback to r+g+b
  float denom = (c > 0) ? (float)c : (float)(r + g + b);
  float red   = denom > 0.0f ? (float)r / denom : 0.0f;
  float green = denom > 0.0f ? (float)g / denom : 0.0f;
  float blue  = denom > 0.0f ? (float)b / denom : 0.0f;

  red = clamp01(red);
  green = clamp01(green);
  blue = clamp01(blue);

  // Phase 3.1: Copy inputs
  input->data.f[0] = red;
  input->data.f[1] = green;
  input->data.f[2] = blue;

  // Phase 3.2: Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed");
    return;
  }

  // Phase 4: Post-processing
  int pred_idx = 0;
  float confidence = 0.0f;

  if (output->type == kTfLiteUInt8) {
    pred_idx = argmax_uint8(output, &confidence);
  } else { // kTfLiteFloat32
    pred_idx = argmax_float(output, &confidence);
  }

  const char* name = kClassNames[pred_idx];
  const char* emoji = kClassEmojis[pred_idx];

  // Print results
  Serial.print("RGBn: ");
  Serial.print(red, 3); Serial.print(", ");
  Serial.print(green, 3); Serial.print(", ");
  Serial.print(blue, 3);
  Serial.print(" -> ");
  Serial.print(name);
  Serial.print(" ");
  Serial.print(emoji);
  Serial.print(" (conf~");
  Serial.print(confidence, 2);
  Serial.println(")");

  // Optional: small delay to stabilize output rate
  // delay handled by loop timer
}