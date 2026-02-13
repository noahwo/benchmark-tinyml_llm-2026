#include <TensorFlowLite.h>  // Must be included before any tflite/micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

#include "model.h"  // Must provide the TFLite flatbuffer, e.g., `const unsigned char model[];`

// Application constants
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// TFLite Micro globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflite_model = nullptr;  // renamed to avoid conflict with model[] from model.h
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 16384;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

static bool initSensor() {
  if (!APDS.begin()) {
    Serial.println("ERROR: APDS9960 initialization failed.");
    return false;
  }
  // No special configuration required for basic RGB reads.
  Serial.println("APDS9960 initialized.");
  return true;
}

static bool acquireNormalizedRGB(float& r_n, float& g_n, float& b_n) {
  // Wait for a new color sample to be available
  const uint32_t start_ms = millis();
  while (!APDS.colorAvailable()) {
    if (millis() - start_ms > 200) {
      return false; // timeout waiting for data
    }
    delay(5);
  }

  int r = 0, g = 0, b = 0;
  APDS.readColor(r, g, b);

  // Normalize to fractions summing to 1.0 (as per dataset)
  const float sum = static_cast<float>(r) + static_cast<float>(g) + static_cast<float>(b);
  if (sum <= 0.0f) {
    return false;
  }

  r_n = static_cast<float>(r) / sum;
  g_n = static_cast<float>(g) / sum;
  b_n = static_cast<float>(b) / sum;

  // Clamp to [0,1] for safety
  r_n = r_n < 0.f ? 0.f : (r_n > 1.f ? 1.f : r_n);
  g_n = g_n < 0.f ? 0.f : (g_n > 1.f ? 1.f : g_n);
  b_n = b_n < 0.f ? 0.f : (b_n > 1.f ? 1.f : b_n);
  return true;
}

void setup() {
  Serial.begin(9600);
  while (!Serial && millis() < 4000) { /* wait for serial */ }

  Serial.println("Object Classifier by Color - TinyML (Nano 33 BLE Sense)");

  // Initialize sensor(s) and peripherals
  if (!initSensor()) {
    Serial.println("Halting due to sensor init failure.");
    while (1) { delay(100); }
  }

  // TFLite Micro setup
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model from model.h (expects symbol like `model` defined there)
  tflite_model = tflite::GetModel(::model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported version %d",
                           tflite_model->version(), TFLITE_SCHEMA_VERSION);
    while (1) { delay(100); }
  }

  // Resolve ops - fallback to AllOpsResolver if unknown
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1) { delay(100); }
  }

  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate tensor properties
  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Unexpected input tensor type. Expected float32.");
    while (1) { delay(100); }
  }
  if (output->type != kTfLiteUInt8) {
    error_reporter->Report("Unexpected output tensor type. Expected uint8.");
    while (1) { delay(100); }
  }

  // Print model I/O info
  Serial.print("Input dims: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print("x");
  }
  Serial.print(", type=float32");
  Serial.println();

  Serial.print("Output dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print("x");
  }
  Serial.print(", type=uint8");
  Serial.println();

  Serial.println("Setup complete. Starting inference...");
}

void loop() {
  // Phase 2: Preprocessing - acquire and normalize RGB
  float r_n = 0.f, g_n = 0.f, b_n = 0.f;
  if (!acquireNormalizedRGB(r_n, g_n, b_n)) {
    // No new sample or invalid reading; try again
    delay(5);
    return;
  }

  // Phase 3: Inference
  // Copy features to input tensor [1,3]
  if (input->bytes < 3 * sizeof(float)) {
    error_reporter->Report("Input tensor too small for 3 features.");
    delay(50);
    return;
  }
  input->data.f[0] = r_n; // 'Red'
  input->data.f[1] = g_n; // 'Green'
  input->data.f[2] = b_n; // 'Blue'

  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    delay(50);
    return;
  }

  // Phase 4: Postprocessing
  // Argmax over 3 classes on uint8 outputs
  int out_elems = 1;
  for (int i = 0; i < output->dims->size; i++) {
    out_elems *= output->dims->data[i];
  }
  if (out_elems < 3) {
    error_reporter->Report("Unexpected output size (%d). Expected >= 3.", out_elems);
    delay(50);
    return;
  }

  const uint8_t* scores = output->data.uint8;
  int best_idx = 0;
  uint8_t best_score = scores[0];
  for (int i = 1; i < 3; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
  }

  // Print result
  Serial.print("RGB(norm): R=");
  Serial.print(r_n, 3);
  Serial.print(" G=");
  Serial.print(g_n, 3);
  Serial.print(" B=");
  Serial.print(b_n, 3);
  Serial.print(" | Class: ");
  Serial.print(kClassNames[best_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[best_idx]);
  Serial.print(" | scores(u8): [");
  Serial.print(scores[0]);
  Serial.print(", ");
  Serial.print(scores[1]);
  Serial.print(", ");
  Serial.print(scores[2]);
  Serial.println("]");

  delay(150);
}