/*
  Object Classifier by Color
  - Arduino Nano 33 BLE Sense + APDS9960 color sensor
  - TensorFlow Lite for Microcontrollers
  - Classifies objects into: Apple, Banana, Orange
  - Prints label with emoji over Serial at 9600 baud
*/

#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"  // Provides TFLITE_SCHEMA_VERSION
#include "model.h"  // Must define the TFLite flatbuffer array, e.g., `const unsigned char model[];`

// ----- TensorFlow Lite Micro globals -----
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  // Use AllOpsResolver for simplicity/compatibility (may be replaced with a minimal resolver if known)
  tflite::AllOpsResolver resolver;

  // Per spec: 20 KB tensor arena
  constexpr int kTensorArenaSize = 20480;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

// ----- Application constants -----
static const char* kClassLabels[3] = { "Apple", "Banana", "Orange" };
static const char* kClassEmojis[3] = { "🍎", "🍌", "🍊" };
static const char* kFallbackLabel = "Unknown";
constexpr uint32_t kSampleIntervalMs = 200;

// ----- Utility functions -----
static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static uint8_t quantizeFloatToUInt8(float value, float scale, int32_t zero_point) {
  // Clamp input to [0,1] domain before quantization if appropriate for RGB inputs
  float q = (value / scale) + static_cast<float>(zero_point);
  int32_t qi = static_cast<int32_t>(q + (q >= 0 ? 0.5f : -0.5f));
  if (qi < 0) qi = 0;
  if (qi > 255) qi = 255;
  return static_cast<uint8_t>(qi);
}

// ----- Setup -----
void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color (APDS9960 + TFLite Micro)");
  Serial.println("Initializing APDS9960 color sensor...");

  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960. Check wiring/power.");
    while (true) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // Load TFLite model from the C array in model.h
  // NOTE: Prior build error used `g_model`; here we use `model` which is the common symbol name.
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema version mismatch: ");
    Serial.print(tfl_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (true) { delay(1000); }
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input sanity check per spec (expect 1x3 features: Red, Green, Blue)
  if (!(input && input->dims && (input->dims->size >= 2))) {
    Serial.println("ERROR: Invalid input tensor.");
    while (true) { delay(1000); }
  }
  int input_features = input->dims->data[input->dims->size - 1];
  if (input_features != 3) {
    Serial.print("WARNING: Expected 3 input features, got ");
    Serial.println(input_features);
  }

  Serial.println("TFLite Micro initialized. Starting inference loop...");
}

// ----- Loop -----
void loop() {
  static uint32_t last_inference_ms = 0;
  uint32_t now = millis();
  if (now - last_inference_ms < kSampleIntervalMs) {
    delay(5);
    return;
  }

  // Wait for new color data
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  // Read raw RGBC values
  int r_raw = 0, g_raw = 0, b_raw = 0, c_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw, c_raw)) {
    // If read fails, skip this cycle
    delay(5);
    return;
  }

  // Preprocessing: normalize by Clear channel and clamp to [0,1]
  // Avoid division by zero by substituting 1 when c_raw <= 0
  int c_safe = (c_raw > 0) ? c_raw : 1;
  float red   = clamp01(static_cast<float>(r_raw) / static_cast<float>(c_safe));
  float green = clamp01(static_cast<float>(g_raw) / static_cast<float>(c_safe));
  float blue  = clamp01(static_cast<float>(b_raw) / static_cast<float>(c_safe));

  // Copy features to model input
  if (input->type == kTfLiteFloat32) {
    input->data.f[0] = red;
    input->data.f[1] = green;
    input->data.f[2] = blue;
  } else if (input->type == kTfLiteUInt8) {
    // Quantize using tensor's scale/zero_point if provided
    float s = (input->params.scale == 0.0f) ? (1.0f / 255.0f) : input->params.scale;
    int32_t zp = input->params.zero_point;
    input->data.uint8[0] = quantizeFloatToUInt8(red,   s, zp);
    input->data.uint8[1] = quantizeFloatToUInt8(green, s, zp);
    input->data.uint8[2] = quantizeFloatToUInt8(blue,  s, zp);
  } else {
    Serial.println("ERROR: Unsupported input tensor type.");
    delay(50);
    return;
  }

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Invoke failed.");
    delay(50);
    return;
  }

  // Postprocessing: argmax over 3 classes
  int best_idx = -1;
  float best_score_f = -1e30f;
  int best_score_u8 = -1;

  if (output->type == kTfLiteFloat32) {
    // Expect 3 logits/scores
    for (int i = 0; i < 3; i++) {
      float v = output->data.f[i];
      if (i == 0 || v > best_score_f) {
        best_score_f = v;
        best_idx = i;
      }
    }
  } else if (output->type == kTfLiteUInt8) {
    for (int i = 0; i < 3; i++) {
      int v = static_cast<int>(output->data.uint8[i]);
      if (i == 0 || v > best_score_u8) {
        best_score_u8 = v;
        best_idx = i;
      }
    }
  } else {
    Serial.println("ERROR: Unsupported output tensor type.");
    delay(50);
    return;
  }

  // Emit result
  const char* label = kFallbackLabel;
  const char* emoji = "?";
  if (best_idx >= 0 && best_idx < 3) {
    label = kClassLabels[best_idx];
    emoji = kClassEmojis[best_idx];
  }

  Serial.print("RGB(norm): ");
  Serial.print(red, 3); Serial.print(", ");
  Serial.print(green, 3); Serial.print(", ");
  Serial.print(blue, 3);
  Serial.print("  ->  ");
  Serial.print(label);
  Serial.print(" ");
  Serial.println(emoji);

  last_inference_ms = now;
}