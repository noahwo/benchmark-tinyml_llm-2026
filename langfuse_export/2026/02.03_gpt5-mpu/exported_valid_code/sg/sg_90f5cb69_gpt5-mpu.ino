#include <Arduino.h>

// Programming Guidelines Phase 1.1: Include Necessary Libraries
#include "TensorFlowLite.h"  // Base TFLM header (must be included before dependent headers)
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino_APDS9960.h>

// Model include (CRITICAL)
#include "model.h"

// If your model header declares these symbols, keep them extern.
// This matches common TFLM Arduino examples (hello_world, micro_speech).
extern const unsigned char g_model[];
extern const int g_model_len;

// Programming Guidelines Phase 1.2: Declare Variables
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflite_model = nullptr;  // renamed to avoid conflict with model[] from model.h
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Programming Guidelines Phase 1.3: Define Tensor Arena
  // Use 16 KB as specified.
  constexpr int kTensorArenaSize = 16384;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  // Application specifics
  const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
  const char* kClassEmojis[3] = {"\xF0\x9F\x8D\x8E", "\xF0\x9F\x8D\x8C", "\xF0\x9F\x8D\x8A"}; // 🍎, 🍌, 🍊 in UTF-8
}

// Helper to compute argmax on uint8/float arrays
int argmax_uint8(const uint8_t* data, int len) {
  int idx = 0;
  uint8_t best = data[0];
  for (int i = 1; i < len; i++) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

int argmax_float(const float* data, int len) {
  int idx = 0;
  float best = data[0];
  for (int i = 1; i < len; i++) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

void setup() {
  // Serial setup
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color - Nano 33 BLE Sense");
  Serial.println("Initializing...");

  // Programming Guidelines Phase 1.9: Initialize other components first if desired
  if (!APDS.begin()) {
    Serial.println("Error: Failed to initialize APDS9960 color sensor.");
  } else {
    Serial.println("APDS9960 sensor initialized.");
  }

  // Programming Guidelines Phase 1.2: Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Programming Guidelines Phase 1.4: Load the Model
  // Use the byte array symbol defined in model.h (model[])
  tflite_model = tflite::GetModel(::model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported version %d.",
                           tflite_model->version(), TFLITE_SCHEMA_VERSION);
    // Do not proceed if versions mismatch
    while (1) { delay(100); }
  }

  // Programming Guidelines Phase 1.5: Resolve Operators
  static tflite::AllOpsResolver resolver;  // Use AllOpsResolver as fallback when operators are unknown

  // Programming Guidelines Phase 1.6: Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Programming Guidelines Phase 1.7: Allocate Memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1) { delay(100); }
  }

  // Programming Guidelines Phase 1.8: Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor: expect [1,3] float32
  bool input_ok = true;
  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Input tensor type mismatch. Expected float32.");
    input_ok = false;
  }
  if (!(input->dims->size == 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    error_reporter->Report("Input tensor shape mismatch. Expected [1,3].");
    input_ok = false;
  }

  // Validate output tensor dims: expect [1,3] and likely uint8 quantized
  bool output_ok = true;
  if (!(output->dims->size == 2 && output->dims->data[0] == 1 && output->dims->data[1] == 3)) {
    error_reporter->Report("Output tensor shape mismatch. Expected [1,3].");
    output_ok = false;
  }
  if (!(output->type == kTfLiteUInt8 || output->type == kTfLiteFloat32)) {
    error_reporter->Report("Output tensor type unexpected. Expected uint8 (quantized) or float32.");
    output_ok = false;
  }

  if (!input_ok || !output_ok) {
    Serial.println("Model IO configuration invalid. Halting.");
    while (1) { delay(100); }
  }

  Serial.println("Initialization complete. Waiting for color data...");
}

void loop() {
  // Programming Guidelines Phase 2.1: Sensor Setup/Data Acquisition
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int rRaw, gRaw, bRaw;
  APDS.readColor(rRaw, gRaw, bRaw);

  // Programming Guidelines Phase 2.2: Optional Feature Extraction
  // Normalize RGB to sum to 1.0 matching dataset scale.
  float r = (float)rRaw;
  float g = (float)gRaw;
  float b = (float)bRaw;
  float sum = r + g + b;
  float rn = 0.0f, gn = 0.0f, bn = 0.0f;
  if (sum > 0.0f) {
    rn = r / sum;
    gn = g / sum;
    bn = b / sum;
  }

  // Programming Guidelines Phase 3.1: Data Copy to Input Tensor
  input->data.f[0] = rn;
  input->data.f[1] = gn;
  input->data.f[2] = bn;

  // Programming Guidelines Phase 3.2: Invoke Interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error: Inference failed.");
    delay(50);
    return;
  }

  // Programming Guidelines Phase 4.1: Process Output
  int num_classes = output->dims->data[1];
  int best_idx = 0;

  if (output->type == kTfLiteUInt8) {
    const uint8_t* scores = output->data.uint8;
    best_idx = argmax_uint8(scores, num_classes);

    // Programming Guidelines Phase 4.2: Execute Application Behavior
    Serial.print("RGB(norm): [");
    Serial.print(rn, 3); Serial.print(", ");
    Serial.print(gn, 3); Serial.print(", ");
    Serial.print(bn, 3); Serial.print("]  ->  Pred: ");
    Serial.print(kClassNames[best_idx]);
    Serial.print(" ");
    Serial.print(kClassEmojis[best_idx]);
    Serial.print("  Scores(u8): [");
    for (int i = 0; i < num_classes; i++) {
      Serial.print((int)scores[i]);
      if (i < num_classes - 1) Serial.print(", ");
    }
    Serial.println("]");
  } else {
    const float* scores = output->data.f;
    best_idx = argmax_float(scores, num_classes);

    Serial.print("RGB(norm): [");
    Serial.print(rn, 3); Serial.print(", ");
    Serial.print(gn, 3); Serial.print(", ");
    Serial.print(bn, 3); Serial.print("]  ->  Pred: ");
    Serial.print(kClassNames[best_idx]);
    Serial.print(" ");
    Serial.print(kClassEmojis[best_idx]);
    Serial.print("  Scores(f32): [");
    for (int i = 0; i < num_classes; i++) {
      Serial.print(scores[i], 4);
      if (i < num_classes - 1) Serial.print(", ");
    }
    Serial.println("]");
  }

  delay(100);
}