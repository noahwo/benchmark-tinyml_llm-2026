/*
  Color-Based Object Classifier
  Board: Arduino Nano 33 BLE Sense
  Sensors: APDS-9960 (RGB)
  Runtime: TensorFlow Lite for Microcontrollers
  Model: included via model.h (uint8_t array named 'model')
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Model array
#include "model.h"

// Application settings
static const uint32_t kBaudRate = 9600;
static const uint32_t kSamplingIntervalMs = 100;
static const int kNumFeatures = 3;    // Red, Green, Blue
static const int kNumClasses = 3;     // Apple, Banana, Orange
static const char* kLabels[kNumClasses] = { "Apple", "Banana", "Orange" };
static const char* kEmojis[kNumClasses] = { "🍎", "🍌", "🍊" };

// Tensor arena
constexpr int kTensorArenaSize = 12288;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// TFLM globals
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static const tflite::Model* tfl_model = nullptr;
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// Timing
static uint32_t last_sample_ms = 0;

static int argmax_uint8(const uint8_t* data, int length) {
  int max_index = 0;
  uint8_t max_value = data[0];
  for (int i = 1; i < length; ++i) {
    if (data[i] > max_value) {
      max_value = data[i];
      max_index = i;
    }
  }
  return max_index;
}

void setup() {
  Serial.begin(kBaudRate);
  while (!Serial) { delay(10); }

  // Initialize sensor
  if (!APDS.begin()) {
    Serial.println("APDS-9960 init failed. Halting.");
    while (true) { delay(100); }
  }

  // Load model from C array 'model' (provided by model.h)
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(tfl_model->version());
    Serial.print(" not equal to supported version ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(100); }
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) { delay(100); }
  }

  // Cache input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input/output checks
  if (input->type != kTfLiteFloat32 || input->dims->size < 2 || input->dims->data[input->dims->size - 1] != kNumFeatures) {
    Serial.println("Unexpected model input tensor shape/type.");
    while (true) { delay(100); }
  }
  if (output->type != kTfLiteUInt8 || output->dims->size < 2 || output->dims->data[output->dims->size - 1] != kNumClasses) {
    Serial.println("Unexpected model output tensor shape/type.");
    while (true) { delay(100); }
  }

  Serial.println("Setup complete. Starting inference...");
}

void loop() {
  const uint32_t now = millis();
  if (now - last_sample_ms < kSamplingIntervalMs) {
    return;
  }
  last_sample_ms = now;

  // Read RGB from APDS-9960
  int r_raw = 0, g_raw = 0, b_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw)) {
    // If not ready, try next cycle
    return;
  }

  // Preprocess: normalize to [0,1]
  float r = constrain(r_raw, 0, 255) / 255.0f;
  float g = constrain(g_raw, 0, 255) / 255.0f;
  float b = constrain(b_raw, 0, 255) / 255.0f;

  // Copy features into input tensor in order [Red, Green, Blue]
  input->data.f[0] = r;
  input->data.f[1] = g;
  input->data.f[2] = b;

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Postprocess: argmax over uint8 scores
  const uint8_t* scores = output->data.uint8;
  int idx = argmax_uint8(scores, kNumClasses);

  // Output: "{emoji} {label}\n"
  Serial.print(kEmojis[idx]);
  Serial.print(" ");
  Serial.println(kLabels[idx]);
}