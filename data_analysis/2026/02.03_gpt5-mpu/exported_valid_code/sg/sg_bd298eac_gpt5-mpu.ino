#include <Arduino.h>

// Phase 1.1: Include Necessary Libraries (TensorFlowLite base first)
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Sensor and I/O libraries
#include <Wire.h>
#include <Arduino_APDS9960.h>

// Model header (must be provided externally)
#include "model.h"

// Phase 1.2: Declare Variables
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflm_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Phase 1.3: Define Tensor Arena
  constexpr int kTensorArenaSize = 4096;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  // App configuration
  constexpr uint32_t SERIAL_BAUD = 9600;
  const char* kClassNames[3]  = {"Apple", "Banana", "Orange"};
  const char* kClassEmojis[3] = {u8"🍎",  u8"🍌",    u8"🍊"};
}

// Utility: safe clamp
static inline float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// Utility: Argmax over N elements of float buffer
int argmax_float(const float* arr, int n) {
  int idx = 0;
  float best = arr[0];
  for (int i = 1; i < n; i++) {
    if (arr[i] > best) {
      best = arr[i];
      idx = i;
    }
  }
  return idx;
}

// Utility: Argmax over N elements of uint8 buffer
int argmax_u8(const uint8_t* arr, int n) {
  int idx = 0;
  uint8_t best = arr[0];
  for (int i = 1; i < n; i++) {
    if (arr[i] > best) {
      best = arr[i];
      idx = i;
    }
  }
  return idx;
}

void setup() {
  // Phase 1.9: Setup Serial
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) {
    ; // wait for serial if connected
  }

  // Phase 1.9: Initialize sensor
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 color sensor.");
    while (1) delay(1000);
  }
  Serial.println("APDS9960 color sensor initialized.");

  // Phase 1.2: Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the Model
  // Expect model to be defined in model.h as a TFLite flatbuffer array.
  tflm_model = tflite::GetModel(model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported version %d.",
                           tflm_model->version(), TFLITE_SCHEMA_VERSION);
    while (1) delay(1000);
  }

  // Phase 1.5: Resolve Operators
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate Memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1) delay(1000);
  }

  // Phase 1.8: Define Model Inputs/Outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Expected float32 input tensor, got type %d", input->type);
    while (1) delay(1000);
  }

  // Input tensor should have at least 3 float elements (shape [1,3] expected)
  int input_elems = input->bytes / sizeof(float);
  if (input_elems < 3) {
    error_reporter->Report("Input tensor too small: %d elements", input_elems);
    while (1) delay(1000);
  }

  Serial.println("TinyML Object Classifier by Color ready.");
  Serial.println("Hold a colored object near the sensor.");
}

void loop() {
  // Phase 2.1: Sensor Setup and Data Acquisition
  int r = 0, g = 0, b = 0, a = 0;

  // Wait until new color data is available
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }
  APDS.readColor(r, g, b, a);

  // Phase 2.2: Preprocessing: normalize to chromaticity (R+G+B=1)
  float rf = static_cast<float>(r);
  float gf = static_cast<float>(g);
  float bf = static_cast<float>(b);
  float sum = rf + gf + bf;
  if (sum < 1.0f) sum = 1.0f; // avoid div by zero

  rf = clampf(rf / sum, 0.0f, 1.0f);
  gf = clampf(gf / sum, 0.0f, 1.0f);
  bf = clampf(bf / sum, 0.0f, 1.0f);

  // Phase 3.1: Copy data to model input
  input->data.f[0] = rf;   // Red
  input->data.f[1] = gf;   // Green
  input->data.f[2] = bf;   // Blue

  // Phase 3.2: Invoke Interpreter
  uint32_t t0 = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    delay(100);
    return;
  }
  uint32_t infer_us = micros() - t0;

  // Phase 4.1: Process Output
  int predicted_idx = 0;

  // We'll compute per-class "probabilities" for printing, handling both uint8 and float models
  float probs[3] = {0, 0, 0};

  if (output->type == kTfLiteUInt8) {
    const uint8_t* out = output->data.uint8;
    predicted_idx = argmax_u8(out, 3);

    // Dequantize for display (optional)
    float scale = output->params.scale;
    int zp = output->params.zero_point;
    for (int i = 0; i < 3; i++) {
      probs[i] = scale * (static_cast<int>(out[i]) - zp);
      // Clamp to [0,1] for nicer display; values may not be calibrated as true probabilities
      probs[i] = clampf(probs[i], 0.0f, 1.0f);
    }
  } else if (output->type == kTfLiteFloat32) {
    const float* out = output->data.f;
    predicted_idx = argmax_float(out, 3);
    for (int i = 0; i < 3; i++) {
      probs[i] = clampf(out[i], 0.0f, 1.0f);
    }
  } else {
    error_reporter->Report("Unsupported output tensor type: %d", output->type);
    delay(200);
    return;
  }

  // Phase 4.2: Execute Application Behavior
  Serial.print("RGB norm: R=");
  Serial.print(rf, 3);
  Serial.print(" G=");
  Serial.print(gf, 3);
  Serial.print(" B=");
  Serial.print(bf, 3);

  Serial.print(" | Inference: ");
  Serial.print(infer_us);
  Serial.print(" us");

  Serial.print(" | Prediction: ");
  Serial.print(kClassNames[predicted_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[predicted_idx]);

  Serial.print(" | Scores: [");
  for (int i = 0; i < 3; i++) {
    Serial.print(probs[i], 3);
    if (i < 2) Serial.print(", ");
  }
  Serial.println("]");

  delay(200);
}