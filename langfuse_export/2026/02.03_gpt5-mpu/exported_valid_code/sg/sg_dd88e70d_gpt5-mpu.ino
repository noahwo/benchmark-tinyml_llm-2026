#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include "model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

// TensorFlow Lite Micro globals
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflm_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Class labels and emojis
const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};
}  // namespace

// Helper to resolve the model data pointer (model is defined in model.h)
const unsigned char* ResolveModelDataPointer() {
  return model;
}

// Read RGB from APDS9960 and normalize to [0, 1]
bool readNormalizedRGB(float& r_norm, float& g_norm, float& b_norm) {
  if (!APDS.colorAvailable()) {
    return false;
  }
  int r = 0, g = 0, b = 0;
  APDS.readColor(r, g, b);
  r_norm = constrain((float)r / 255.0f, 0.0f, 1.0f);
  g_norm = constrain((float)g / 255.0f, 0.0f, 1.0f);
  b_norm = constrain((float)b / 255.0f, 0.0f, 1.0f);
  return true;
}

// Argmax utility
int argmax(const float* arr, int n) {
  int idx = 0;
  float best = arr[0];
  for (int i = 1; i < n; ++i) {
    if (arr[i] > best) {
      best = arr[i];
      idx = i;
    }
  }
  return idx;
}

void setup() {
  Serial.begin(9600);
  while (!Serial) { /* wait for serial */ }

  // Initialize sensor
  if (!APDS.begin()) {
    Serial.println("Error: Failed to initialize APDS9960 sensor.");
    while (1) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // Optional: Initialize BLE (not used for this demo, but included per dependencies)
  if (!BLE.begin()) {
    Serial.println("Warning: BLE init failed. Continuing without BLE.");
  } else {
    BLE.end(); // Not used; stop to save power.
  }

  // Set up TFLite Micro error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Resolve and load the model
  const unsigned char* model_data = ResolveModelDataPointer();
  if (model_data == nullptr) {
    Serial.println("Error: Could not find model data in model.h.");
    while (1) { delay(1000); }
  }

  tflm_model = tflite::GetModel(model_data);
#ifdef TFLITE_SCHEMA_VERSION
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Error: Model schema mismatch. Model schema: ");
    Serial.print(tflm_model->version());
    Serial.print(" != TFLite Schema: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }
#endif

  // Use AllOpsResolver as a safe default when operator set is unknown
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Error: AllocateTensors() failed.");
    while (1) { delay(1000); }
  }

  // Retrieve input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input validation
  if (input->type != kTfLiteFloat32) {
    Serial.println("Error: Expected float32 input tensor.");
    while (1) { delay(1000); }
  }
  if (!(input->dims->size >= 2 && input->dims->data[input->dims->size - 1] == 3)) {
    Serial.println("Error: Expected input shape [1,3] (or last dim=3).");
    while (1) { delay(1000); }
  }

  Serial.println("TinyML Object Classifier by Color is ready.");
}

void loop() {
  // Acquire and preprocess sensor data
  float r = 0.0f, g = 0.0f, b = 0.0f;
  if (!readNormalizedRGB(r, g, b)) {
    delay(5);
    return;
  }

  // Copy to model input tensor
  input->data.f[0] = r;
  input->data.f[1] = g;
  input->data.f[2] = b;

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error: Inference failed.");
    delay(100);
    return;
  }

  // Postprocess output
  const int out_dims = (output->dims->size >= 2) ? output->dims->data[output->dims->size - 1] : output->dims->data[0];
  const int num_classes = min(out_dims, 3); // Expecting 3 classes: Apple, Banana, Orange

  float scores[3] = {0.f, 0.f, 0.f};
  if (output->type == kTfLiteUInt8) {
    const float scale = output->params.scale;
    const int zero_point = output->params.zero_point;
    for (int i = 0; i < num_classes; ++i) {
      uint8_t v = output->data.uint8[i];
      scores[i] = scale * (static_cast<int>(v) - zero_point);
    }
  } else if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < num_classes; ++i) {
      scores[i] = output->data.f[i];
    }
  } else {
    Serial.println("Error: Unsupported output tensor type.");
    delay(100);
    return;
  }

  int idx = argmax(scores, num_classes);

  // Print result to Serial with emoji
  Serial.print("RGB(norm) = [");
  Serial.print(r, 3); Serial.print(", ");
  Serial.print(g, 3); Serial.print(", ");
  Serial.print(b, 3); Serial.print("] -> ");

  if (num_classes == 3) {
    Serial.print("Class: ");
    Serial.print(kClassNames[idx]);
    Serial.print(" ");
    Serial.print(kClassEmojis[idx]);
    Serial.print(" | Scores: ");
    Serial.print(scores[0], 3); Serial.print(", ");
    Serial.print(scores[1], 3); Serial.print(", ");
    Serial.println(scores[2], 3);
  } else {
    Serial.print("Pred index: ");
    Serial.print(idx);
    Serial.print(" | Scores: ");
    for (int i = 0; i < out_dims; ++i) {
      Serial.print(scores[i], 3);
      if (i < out_dims - 1) Serial.print(", ");
    }
    Serial.println();
  }

  delay(100);
}