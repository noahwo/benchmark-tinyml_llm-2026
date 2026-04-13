#include <Arduino.h>
#include <TensorFlowLite.h>  // Base TFLite Micro library (must come before dependent headers)
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <SPI.h>
#include <ArduinoBLE.h>

#include "model.h"  // Must provide model[] flatbuffer array

// Application/Model configuration
static const int kSerialBaud = 9600;
static const uint32_t kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// TFLite Micro globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Classification labels and emojis
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {
  "\xF0\x9F\x8D\x8E", // 🍎
  "\xF0\x9F\x8D\x8C", // 🍌
  "\xF0\x9F\x8D\x8A"  // 🍊
};

void printTensorInfo(const TfLiteTensor* t, const char* name) {
  Serial.print(name);
  Serial.print(" -> type: ");
  Serial.print(t->type);
  Serial.print(", dims: [");
  for (int i = 0; i < t->dims->size; i++) {
    Serial.print(t->dims->data[i]);
    if (i < t->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
}

void setup() {
  Serial.begin(kSerialBaud);
  while (!Serial && millis() < 4000) {
    delay(10);
  }
  Serial.println("Object Classifier by Color - Starting");

  // Initialize RGB color sensor
  if (!APDS.begin()) {
    Serial.println("Error: Failed to initialize APDS9960 sensor.");
    while (1) {
      delay(1000);
    }
  }
  // Optional: Give the sensor a moment to stabilize
  delay(50);

  // TFLite Micro error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model from model.h (expects model[] symbol)
  tfl_model = tflite::GetModel(::model);
  // Guarded schema version check to support library variants
  #ifdef TFLITE_SCHEMA_VERSION
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema version mismatch. Found: ");
    Serial.print(tfl_model->version());
    Serial.print(", expected: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) {
      delay(1000);
    }
  }
  #endif

  // Operator resolver (AllOps as a safe default)
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("Error: AllocateTensors() failed");
    while (1) {
      delay(1000);
    }
  }

  // Obtain input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor: expected [1, 3], float32
  bool input_ok = true;
  if (input->type != kTfLiteFloat32) {
    Serial.println("Warning: Input tensor is not float32 as expected.");
    input_ok = false;
  }
  if (!(input->dims->size == 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.println("Warning: Input tensor shape is not [1, 3] as expected.");
    input_ok = false;
  }

  // Validate output tensor: expected [1, 3], uint8 (quantized) or float32
  bool output_ok = true;
  if (output->type != kTfLiteUInt8 && output->type != kTfLiteFloat32) {
    Serial.println("Warning: Output tensor type is neither uint8 nor float32.");
    output_ok = false;
  }
  if (!(output->dims->size == 2 && output->dims->data[0] == 1 && output->dims->data[1] == 3)) {
    Serial.println("Warning: Output tensor shape is not [1, 3] as expected.");
    output_ok = false;
  }

  // Debug tensor info
  printTensorInfo(input, "Input");
  printTensorInfo(output, "Output");

  if (!input_ok || !output_ok) {
    Serial.println("Tensor configuration mismatch. Attempting to continue, but results may be invalid.");
  }

  Serial.println("Initialization complete.");
}

void loop() {
  // Wait for a new color reading
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int r_raw, g_raw, b_raw;
  // readColor() in Arduino_APDS9960 returns void; call without checking return
  APDS.readColor(r_raw, g_raw, b_raw);

  // Normalize RGB to sum to 1 (matching dataset characteristics)
  float r = (float)max(r_raw, 0);
  float g = (float)max(g_raw, 0);
  float b = (float)max(b_raw, 0);
  float sum = r + g + b;
  float rn = 0.0f, gn = 0.0f, bn = 0.0f;
  if (sum > 0.0f) {
    rn = r / sum;
    gn = g / sum;
    bn = b / sum;
  }

  // Copy to input tensor [1, 3]
  if (input && input->type == kTfLiteFloat32 && input->dims->size >= 2) {
    input->data.f[0] = rn;
    input->data.f[1] = gn;
    input->data.f[2] = bn;
  } else {
    // If input type unexpected, cannot proceed safely
    Serial.println("Error: Invalid input tensor configuration.");
    delay(200);
    return;
  }

  // Invoke inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error: Inference failed.");
    delay(200);
    return;
  }

  // Process output
  int best_idx = 0;
  float scores[3] = {0, 0, 0};

  if (output->type == kTfLiteUInt8) {
    // Quantized output: [0..255]
    uint8_t s0 = output->data.uint8[0];
    uint8_t s1 = output->data.uint8[1];
    uint8_t s2 = output->data.uint8[2];
    scores[0] = s0 / 255.0f;
    scores[1] = s1 / 255.0f;
    scores[2] = s2 / 255.0f;
  } else if (output->type == kTfLiteFloat32) {
    scores[0] = output->data.f[0];
    scores[1] = output->data.f[1];
    scores[2] = output->data.f[2];
  } else {
    Serial.println("Error: Unsupported output tensor type.");
    delay(200);
    return;
  }

  // Argmax
  best_idx = 0;
  float best_score = scores[0];
  for (int i = 1; i < 3; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
  }

  // Output to Serial
  Serial.print("Input RGB(norm): ");
  Serial.print(rn, 3); Serial.print(", ");
  Serial.print(gn, 3); Serial.print(", ");
  Serial.print(bn, 3);
  Serial.print(" | Prediction: ");
  Serial.print(kClassNames[best_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[best_idx]);
  Serial.print(" | Scores [%]: [");
  Serial.print(scores[0] * 100.0f, 1); Serial.print(", ");
  Serial.print(scores[1] * 100.0f, 1); Serial.print(", ");
  Serial.print(scores[2] * 100.0f, 1); Serial.println("]");

  delay(150);
}