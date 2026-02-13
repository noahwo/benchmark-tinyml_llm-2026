#include <Arduino.h>

// Phase 1.1: Include necessary libraries (base TensorFlowLite first)
#include <TensorFlowLite.h>
#include "model.h"
#include <Arduino_APDS9960.h>
#include <Wire.h>

// TFLite Micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Phase 1.2: Declare critical TFLM variables
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Phase 1.3: Tensor arena
constexpr int kTensorArenaSize = 16384;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// Application configuration
static const uint32_t kBaudRate = 9600;
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};  // Unicode emojis

// Helper: Acquire and normalize RGB as fractions that sum to 1.0
bool readNormalizedRGB(float& r_norm, float& g_norm, float& b_norm) {
  int r = 0, g = 0, b = 0, a = 0;

  // Wait for new data
  if (!APDS.colorAvailable()) {
    return false;
  }

  // Read raw color
  APDS.readColor(r, g, b, a);

  // Normalize using RGB sum (dataset appears to use normalized RGB)
  const int sum = r + g + b;
  if (sum <= 0) {
    return false;
  }

  r_norm = static_cast<float>(r) / static_cast<float>(sum);
  g_norm = static_cast<float>(g) / static_cast<float>(sum);
  b_norm = static_cast<float>(b) / static_cast<float>(sum);

  return true;
}

// Helper: Get dequantized output value as float for comparison
float get_output_value(const TfLiteTensor* out, int idx) {
  if (out->type == kTfLiteUInt8) {
    const float scale = out->params.scale;
    const int zp = out->params.zero_point;
    const int32_t q = static_cast<int32_t>(out->data.uint8[idx]);
    return (static_cast<float>(q) - static_cast<float>(zp)) * scale;
  } else if (out->type == kTfLiteFloat32) {
    return out->data.f[idx];
  } else if (out->type == kTfLiteInt8) {
    const float scale = out->params.scale;
    const int zp = out->params.zero_point;
    const int32_t q = static_cast<int32_t>(out->data.int8[idx]);
    return (static_cast<float>(q) - static_cast<float>(zp)) * scale;
  }
  // Unsupported type: return 0
  return 0.0f;
}

void setup() {
  // Phase 1.9: Setup Serial
  Serial.begin(kBaudRate);
  while (!Serial) { delay(10); }
  Serial.println("Object Classifier by Color - TinyML");

  // Phase 1.9: Initialize sensor
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor.");
    while (true) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // Phase 1.2: Initialize error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load model
  tflite_model = tflite::GetModel(::model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(tflite_model->version());
    Serial.print(" is not equal to supported schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Phase 1.5: Resolve operators
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (true) { delay(1000); }
  }

  // Phase 1.8: Define model IO
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor
  if (input->type != kTfLiteFloat32) {
    Serial.println("ERROR: Model input tensor must be float32.");
    while (true) { delay(1000); }
  }
  if (!(input->dims->size >= 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.println("ERROR: Model input shape must be [1, 3].");
    while (true) { delay(1000); }
  }

  // Validate output tensor (expected uint8 per spec)
  if (!(output->type == kTfLiteUInt8 || output->type == kTfLiteFloat32 || output->type == kTfLiteInt8)) {
    Serial.println("ERROR: Unsupported output tensor type.");
    while (true) { delay(1000); }
  }
  Serial.println("Model initialized. Starting inference...");
}

void loop() {
  // Phase 2.1/2.2: Sensor read and preprocessing
  float r_n = 0.0f, g_n = 0.0f, b_n = 0.0f;
  if (!readNormalizedRGB(r_n, g_n, b_n)) {
    delay(5);
    return;
  }

  // Clamp to [0,1] to avoid any numerical issues
  r_n = constrain(r_n, 0.0f, 1.0f);
  g_n = constrain(g_n, 0.0f, 1.0f);
  b_n = constrain(b_n, 0.0f, 1.0f);

  // Phase 3.1: Copy data to input tensor
  input->data.f[0] = r_n;
  input->data.f[1] = g_n;
  input->data.f[2] = b_n;

  // Phase 3.2: Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(100);
    return;
  }

  // Phase 4.1: Process output (argmax over 3 classes)
  float scores[3];
  for (int i = 0; i < 3; ++i) {
    scores[i] = get_output_value(output, i);
  }

  int best_idx = 0;
  float best_val = scores[0];
  for (int i = 1; i < 3; ++i) {
    if (scores[i] > best_val) {
      best_val = scores[i];
      best_idx = i;
    }
  }

  // Phase 4.2: Execute application behavior (Serial output with emoji)
  Serial.print("RGB(norm) -> R: ");
  Serial.print(r_n, 3);
  Serial.print(" G: ");
  Serial.print(g_n, 3);
  Serial.print(" B: ");
  Serial.print(b_n, 3);
  Serial.print(" | Scores: ");
  Serial.print(scores[0], 3); Serial.print(", ");
  Serial.print(scores[1], 3); Serial.print(", ");
  Serial.print(scores[2], 3);
  Serial.print(" | Class: ");
  Serial.print(kClassNames[best_idx]);
  Serial.print(" ");
  Serial.println(kClassEmojis[best_idx]);

  delay(150);
}