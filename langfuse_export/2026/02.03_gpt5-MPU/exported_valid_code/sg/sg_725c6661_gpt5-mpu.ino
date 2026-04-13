#include <Arduino.h>
#include "model.h"  // Must provide the flatbuffer model data
#include <Arduino_APDS9960.h>

// Programming Guidelines Phase 1: Include TensorFlow Lite Micro in correct order
#include <TensorFlowLite.h>  // Base library MUST come first
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// TinyML globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflm_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor Arena (as per spec)
  constexpr int kTensorArenaSize = 16384;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// Application configuration
static const unsigned long kSerialBaud = 9600;
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// Utility: Safe argmax for uint8 or float32 buffers
int ArgMaxUint8(const uint8_t* data, int len) {
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

int ArgMaxFloat(const float* data, int len) {
  int idx = 0;
  float best = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

void setup() {
  // Phase 1: Initialization
  Serial.begin(kSerialBaud);
  while (!Serial) { delay(5); }

  // Initialize Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model (model.h must define model buffer)
  tflm_model = tflite::GetModel(model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided schema %d != %d supported.",
                           tflm_model->version(), TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Resolve ops
  static tflite::AllOpsResolver resolver;

  // Instantiate interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // Define model inputs/outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor: expect [1,3] float32
  bool input_ok = (input != nullptr) &&
                  (input->type == kTfLiteFloat32) &&
                  (input->dims != nullptr) &&
                  (input->dims->size == 2) &&
                  (input->dims->data[0] == 1) &&
                  (input->dims->data[1] == 3);
  if (!input_ok) {
    error_reporter->Report("Unexpected input tensor. Expect [1,3] float32.");
    while (true) { delay(1000); }
  }

  // Validate output tensor: expect [1,3] uint8 or float32
  bool output_ok = (output != nullptr) &&
                   (output->dims != nullptr) &&
                   (output->dims->size == 2) &&
                   (output->dims->data[0] == 1) &&
                   (output->dims->data[1] == 3) &&
                   (output->type == kTfLiteUInt8 || output->type == kTfLiteFloat32);
  if (!output_ok) {
    error_reporter->Report("Unexpected output tensor. Expect [1,3] uint8 or float32.");
    while (true) { delay(1000); }
  }

  // Phase 1.9: Initialize sensor
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 color sensor.");
    while (true) { delay(1000); }
  }

  Serial.println("Object Classifier by Color - Ready");
}

void loop() {
  // Phase 2: Preprocessing - Sensor Setup and Feature Extraction
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw)) {
    // Failed to read; skip this cycle
    return;
  }

  // Normalize RGB to fractions that sum to 1.0 (consistent with dataset)
  float r = static_cast<float>(r_raw);
  float g = static_cast<float>(g_raw);
  float b = static_cast<float>(b_raw);
  float sum = r + g + b;

  if (sum <= 0.0f) {
    // Avoid division by zero
    return;
  }

  float r_n = r / sum;
  float g_n = g / sum;
  float b_n = b / sum;

  // Optional: clamp to [0,1] to handle any anomalies
  r_n = r_n < 0.f ? 0.f : (r_n > 1.f ? 1.f : r_n);
  g_n = g_n < 0.f ? 0.f : (g_n > 1.f ? 1.f : g_n);
  b_n = b_n < 0.f ? 0.f : (b_n > 1.f ? 1.f : b_n);

  // Phase 3.1: Copy data into input tensor
  input->data.f[0] = r_n;
  input->data.f[1] = g_n;
  input->data.f[2] = b_n;

  // Phase 3.2: Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Phase 4: Postprocessing - Interpret output
  int num_classes = output->dims->data[1];
  int predicted_index = 0;

  if (output->type == kTfLiteUInt8) {
    const uint8_t* out = output->data.uint8;
    predicted_index = ArgMaxUint8(out, num_classes);
  } else { // kTfLiteFloat32 fallback
    const float* out = output->data.f;
    predicted_index = ArgMaxFloat(out, num_classes);
  }

  // Emit results to Serial with Unicode emojis
  Serial.print("RGB(norm): [");
  Serial.print(r_n, 3);
  Serial.print(", ");
  Serial.print(g_n, 3);
  Serial.print(", ");
  Serial.print(b_n, 3);
  Serial.print("] -> ");

  // Safe guard for index range
  if (predicted_index < 0 || predicted_index >= 3) {
    Serial.println("Prediction index out of range");
  } else {
    Serial.print(kClassNames[predicted_index]);
    Serial.print(" ");
    Serial.println(kClassEmojis[predicted_index]);
  }

  delay(150);
}