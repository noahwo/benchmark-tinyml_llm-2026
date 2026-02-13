#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <ArduinoBLE.h>

// IMPORTANT: Include TensorFlowLite base before dependent headers
#include <TensorFlowLite.h>
#include "model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// TensorFlow Lite Micro globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflm_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 8192;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// App constants
static const char* kClassNames[3] = { "Apple", "Banana", "Orange" };
static const char* kClassEmojis[3] = { "🍎", "🍌", "🍊" };

void setup() {
  // Communication
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color - Nano 33 BLE Sense");
  Serial.println("Initializing APDS9960 color sensor...");

  // Sensor init
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor.");
    while (1) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // TFLite Micro setup
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  tflm_model = tflite::GetModel(::model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema (%d) not equal to supported (%d).",
                           tflm_model->version(), TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Check input expectations
  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Unexpected input type. Expected float32.");
    while (1) { delay(1000); }
  }
  // Optional: print tensor info
  Serial.print("Model input dims: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print("x");
  }
  Serial.print(", type: "); Serial.println("float32");

  Serial.print("Model output dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print("x");
  }
  Serial.print(", type: ");
  if (output->type == kTfLiteUInt8) Serial.println("uint8");
  else if (output->type == kTfLiteFloat32) Serial.println("float32");
  else Serial.println("other");

  Serial.println("Setup complete. Starting inference loop...");
}

void loop() {
  // Wait for color data
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int r = 0, g = 0, b = 0, c = 0;
  APDS.readColor(r, g, b, c);

  // Normalize to ratios that sum to 1, matching dataset characteristics
  float rf = static_cast<float>(r);
  float gf = static_cast<float>(g);
  float bf = static_cast<float>(b);
  float sum = rf + gf + bf;

  if (sum <= 0.0f) {
    // Too dark or invalid reading
    Serial.println("Sensor reading too low; skipping inference.");
    delay(50);
    return;
  }

  float red_ratio   = rf / sum;
  float green_ratio = gf / sum;
  float blue_ratio  = bf / sum;

  // Copy to model input [1, 3] float32
  input->data.f[0] = red_ratio;
  input->data.f[1] = green_ratio;
  input->data.f[2] = blue_ratio;

  // Inference
  uint32_t t0 = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(100);
    return;
  }
  uint32_t dt = micros() - t0;

  // Process output
  int best_idx = -1;
  float best_score = -1.0f;
  float scores[3] = {0, 0, 0};

  if (output->type == kTfLiteUInt8) {
    // Quantized probabilities [0..255]
    for (int i = 0; i < 3; i++) {
      uint8_t s = output->data.uint8[i];
      scores[i] = static_cast<float>(s) / 255.0f;
      if (scores[i] > best_score) {
        best_score = scores[i];
        best_idx = i;
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < 3; i++) {
      scores[i] = output->data.f[i];
      if (scores[i] > best_score) {
        best_score = scores[i];
        best_idx = i;
      }
    }
  } else {
    Serial.println("Unsupported output tensor type.");
    delay(100);
    return;
  }

  // Output result over Serial using emojis
  Serial.print("RGB raw: ");
  Serial.print(r); Serial.print(", ");
  Serial.print(g); Serial.print(", ");
  Serial.print(b);
  Serial.print(" | ratios: R=");
  Serial.print(red_ratio, 3); Serial.print(" G=");
  Serial.print(green_ratio, 3); Serial.print(" B=");
  Serial.print(blue_ratio, 3);

  Serial.print(" | Detected: ");
  if (best_idx >= 0 && best_idx < 3) {
    Serial.print(kClassNames[best_idx]);
    Serial.print(" ");
    Serial.print(kClassEmojis[best_idx]);
  } else {
    Serial.print("Unknown");
  }

  Serial.print(" | scores: ");
  for (int i = 0; i < 3; i++) {
    Serial.print(kClassNames[i]);
    Serial.print("=");
    Serial.print(scores[i] * 100.0f, 1);
    Serial.print("%");
    if (i < 2) Serial.print(", ");
  }

  Serial.print(" | ");
  Serial.print("t_infer=");
  Serial.print(dt);
  Serial.println("us");

  delay(100);
}