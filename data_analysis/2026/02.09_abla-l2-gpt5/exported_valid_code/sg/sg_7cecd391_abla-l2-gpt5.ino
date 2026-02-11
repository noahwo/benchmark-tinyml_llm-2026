/*
  Project: Color-based Object Classifier
  Board:   Arduino Nano 33 BLE Sense
  Sensor:  APDS-9960 (RGB)
  Model:   TensorFlow Lite Micro (included via model.h)

  Behavior:
  - Reads RGB from APDS-9960
  - Normalizes to ratios r=R/(R+G+B) etc.
  - Runs TFLM inference (input: float32[3], output: uint8[3])
  - Prints predicted label with emoji and confidence via Serial
*/

#include <Arduino.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <Arduino_APDS9960.h>
#include "model.h"

// ----- Configuration -----
static const uint32_t kSerialBaud = 9600;
static const int kNumClasses = 3;
static const char* kLabels[kNumClasses] = { "Apple", "Banana", "Orange" };
static const char* kEmojis[kNumClasses] = { "🍎", "🍌", "🍊" };
static const size_t kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// ----- TFLM Globals -----
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static const tflite::Model* tflm_model = nullptr;
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// ----- Helpers -----
static int argmax_u8(const uint8_t* data, int len) {
  int idx = 0;
  uint8_t best = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

static float clip01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

void setup() {
  Serial.begin(kSerialBaud);
  while (!Serial) { delay(10); }

  Serial.println(F("Color-based Object Classifier (Nano 33 BLE Sense)"));
  Serial.println(F("Initializing APDS-9960 color sensor..."));

  if (!APDS.begin()) {
    Serial.println(F("ERROR: APDS-9960 not found. Check wiring or board selection."));
    while (true) { delay(1000); }
  }
  Serial.println(F("APDS-9960 initialized."));

  Serial.println(F("Loading TFLite Micro model..."));
  tflm_model = tflite::GetModel(model);
  if (tflm_model == nullptr) {
    Serial.println(F("ERROR: Failed to get TFLite model."));
    while (true) { delay(1000); }
  }

  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println(F("ERROR: AllocateTensors() failed."));
    while (true) { delay(1000); }
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Basic input/output checks
  if (input_tensor->type != kTfLiteFloat32 || input_tensor->bytes < 3 * sizeof(float)) {
    Serial.println(F("WARNING: Input tensor is not float32[3] as expected."));
  }
  if (output_tensor->type != kTfLiteUInt8 || output_tensor->bytes < kNumClasses * sizeof(uint8_t)) {
    Serial.println(F("WARNING: Output tensor is not uint8[3] as expected."));
  }

  Serial.println(F("Setup complete. Starting inference loop...\n"));
}

void loop() {
  // 1) Read raw RGB from sensor
  int r_raw = 0, g_raw = 0, b_raw = 0, c_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw, c_raw)) {
    // If read failed, wait and try again
    delay(50);
    return;
  }

  // 2) Preprocessing: RGB ratio normalization with clipping
  float R = (float)r_raw;
  float G = (float)g_raw;
  float B = (float)b_raw;
  float sum = R + G + B;
  float r = 0.0f, g = 0.0f, b = 0.0f;
  if (sum > 0.0f) {
    r = clip01(R / sum);
    g = clip01(G / sum);
    b = clip01(B / sum);
  }

  // 3) Copy to input tensor
  float* in = interpreter->typed_input_tensor<float>(0);
  in[0] = r;  // Red
  in[1] = g;  // Green
  in[2] = b;  // Blue

  // 4) Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("ERROR: Invoke failed."));
    delay(100);
    return;
  }

  // 5) Postprocessing: argmax on uint8 scores and map to confidence
  const uint8_t* scores = output_tensor->data.uint8;
  int best_idx = argmax_u8(scores, kNumClasses);
  float confidence = scores[best_idx] / 255.0f;

  // 6) Output result
  Serial.print(F("RGB ratios: ["));
  Serial.print(r, 3); Serial.print(F(", "));
  Serial.print(g, 3); Serial.print(F(", "));
  Serial.print(b, 3); Serial.print(F("]  ->  "));

  Serial.print(kLabels[best_idx]);
  Serial.print(F(" "));
  Serial.print(kEmojis[best_idx]);
  Serial.print(F("  (conf="));
  Serial.print(confidence, 2);
  Serial.println(F(")"));

  delay(150);
}