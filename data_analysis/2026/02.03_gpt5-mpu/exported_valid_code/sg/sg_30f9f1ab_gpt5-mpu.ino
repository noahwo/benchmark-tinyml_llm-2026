#include <TensorFlowLite.h>  // Base TFLite Micro library (must be included before dependent headers)
#include "model.h"           // Model flatbuffer (provides `model` byte array)

#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <ArduinoBLE.h>

// TFLite Micro dependent headers
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// -----------------------------
// Global TFLite Micro variables
// -----------------------------
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflite_model = nullptr;   // Renamed to avoid conflict with model array from model.h
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor Arena
  constexpr int kTensorArenaSize = 16384;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// -----------------------------
// Application configuration
// -----------------------------
static const unsigned long kBaudRate = 9600;
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmoji[3] = {u8"🍎", u8"🍌", u8"🍊"};

// Utility: Get index of max value in a uint8_t array
int argmax_u8(const uint8_t* data, int len) {
  int max_idx = 0;
  uint8_t max_val = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }
  return max_idx;
}

void setup() {
  // -----------------------------
  // Phase 1: Initialization
  // -----------------------------
  Serial.begin(kBaudRate);
  while (!Serial && millis() < 4000) {
    ; // Wait for Serial to be ready (with timeout)
  }

  // Initialize sensor I2C and APDS9960 color sensor
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 color sensor.");
    while (1) { delay(100); }
  }
  Serial.println("APDS9960 initialized.");

  // Optional: Initialize BLE (not used directly, but included per spec)
  // BLE.begin(); // Uncomment if BLE functionality is needed

  // TFLite Micro error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load the model from the byte array `model` defined in model.h
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported schema %d.",
                           tflite_model->version(), TFLITE_SCHEMA_VERSION);
    Serial.println("ERROR: Incompatible TFLite model schema.");
    while (1) { delay(100); }
  }

  // Operator resolver (use AllOpsResolver as fallback when ops are unknown)
  static tflite::AllOpsResolver resolver;

  // Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensor arena
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (1) { delay(100); }
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Check input tensor
  if (input->type != kTfLiteFloat32) {
    Serial.println("ERROR: Model input tensor is not float32.");
    while (1) { delay(100); }
  }
  if (!(input->dims->size == 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.println("WARNING: Model input dims are not [1, 3]. Proceeding anyway.");
  }

  // Check output tensor
  if (output->type != kTfLiteUInt8) {
    Serial.println("WARNING: Model output tensor is not uint8. Proceeding anyway.");
  }
  if (!(output->dims->size == 2 && output->dims->data[0] == 1 && output->dims->data[1] == 3)) {
    Serial.println("WARNING: Model output dims are not [1, 3]. Proceeding anyway.");
  }

  Serial.println("TFLite Micro initialized.");
  Serial.println("Object Classifier by Color is ready.");
  Serial.println("Columns: Red, Green, Blue normalized to sum=1. Output: Apple 🍎, Banana 🍌, Orange 🍊");
}

void loop() {
  // -----------------------------
  // Phase 2: Preprocessing
  // -----------------------------
  // Wait until a color sample is available
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  APDS.readColor(r_raw, g_raw, b_raw);

  // Normalize RGB to sum = 1 (matching dataset distribution)
  float r = static_cast<float>(r_raw);
  float g = static_cast<float>(g_raw);
  float b = static_cast<float>(b_raw);

  float sum = r + g + b;
  float red_n = 0.0f, green_n = 0.0f, blue_n = 0.0f;
  if (sum > 0.0f) {
    red_n = r / sum;
    green_n = g / sum;
    blue_n = b / sum;
  }

  // Optional: simple clamp to [0,1] safety
  if (red_n < 0) red_n = 0; if (red_n > 1) red_n = 1;
  if (green_n < 0) green_n = 0; if (green_n > 1) green_n = 1;
  if (blue_n < 0) blue_n = 0; if (blue_n > 1) blue_n = 1;

  // -----------------------------
  // Phase 3: Inference
  // -----------------------------
  // Copy input to tensor [1,3] float32
  input->data.f[0] = red_n;
  input->data.f[1] = green_n;
  input->data.f[2] = blue_n;

  // Invoke the interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(50);
    return;
  }

  // -----------------------------
  // Phase 4: Postprocessing
  // -----------------------------
  // Interpret output [1,3] uint8 as class scores and pick argmax
  const uint8_t* scores = output->data.uint8;
  int pred_idx = argmax_u8(scores, 3);

  // Emit result
  Serial.print("Input (norm RGB): ");
  Serial.print(red_n, 3); Serial.print(", ");
  Serial.print(green_n, 3); Serial.print(", ");
  Serial.print(blue_n, 3);

  Serial.print("  | Scores: [");
  Serial.print(scores[0]); Serial.print(", ");
  Serial.print(scores[1]); Serial.print(", ");
  Serial.print(scores[2]); Serial.print("]");

  Serial.print("  => Predicted: ");
  Serial.print(kClassNames[pred_idx]);
  Serial.print(" ");
  Serial.println(kClassEmoji[pred_idx]);

  delay(150);
}