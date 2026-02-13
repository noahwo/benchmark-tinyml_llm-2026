#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "model.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Labels and emojis
static const char* kLabels[3] = {"Apple", "Banana", "Orange"};
static const char* kEmojis[3] = {"🍎", "🍌", "🍊"};

// TFLM globals
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* tfl_model = nullptr;

  // Adjust count upward if your model requires additional ops
  tflite::MicroMutableOpResolver<20> resolver;

  constexpr int kTensorArenaSize = 16384;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter* interpreter = nullptr;

  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

// Helper: clamp float to [0,1]
static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

void setup() {
  // Begin Serial at 9600 baud
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Starting RGB classifier (TensorFlow Lite Micro)");

  // Initialize APDS-9960 color sensor
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS-9960!");
    while (1) { delay(100); }
  }
  Serial.println("APDS-9960 initialized.");

  // Create TFLite model from model.h
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(tfl_model->version());
    Serial.print(" is not equal to supported schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(100); }
  }

  // Configure MicroMutableOpResolver and Interpreter
  // Add a common set of ops typically used by small classifiers.
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddRelu();
  resolver.AddRelu6();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddSub();
  resolver.AddLogistic();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors with tensor arena
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1) { delay(100); }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Run one warm-up inference
  if (input && input->type == kTfLiteFloat32 && input->bytes >= (3 * sizeof(float))) {
    input->data.f[0] = 0.0f;
    input->data.f[1] = 0.0f;
    input->data.f[2] = 0.0f;
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Warm-up inference failed");
    while (1) { delay(100); }
  }

  Serial.println("Setup complete.");
}

void loop() {
  // Read R,G,B from APDS-9960
  int r = 0, g = 0, b = 0;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b);
  } else {
    // If no new data, wait briefly and try again
    delay(5);
    return;
  }

  // Normalize to 0..1 float and write to input tensor (order: R, G, B)
  if (!input || input->type != kTfLiteFloat32) {
    Serial.println("Invalid model input tensor");
    delay(200);
    return;
  }
  const float rf = clamp01((float)r / 255.0f);
  const float gf = clamp01((float)g / 255.0f);
  const float bf = clamp01((float)b / 255.0f);

  input->data.f[0] = rf;
  input->data.f[1] = gf;
  input->data.f[2] = bf;

  // Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed");
    delay(200);
    return;
  }

  // Read uint8 class index from output tensor
  if (!output || output->type != kTfLiteUInt8 || output->bytes < 1) {
    Serial.println("Invalid model output tensor");
    delay(200);
    return;
  }
  uint8_t idx = output->data.uint8[0];

  // Map index to class label and emoji
  const char* label = (idx < 3) ? kLabels[idx] : "Unknown";
  const char* emoji = (idx < 3) ? kEmojis[idx] : "❓";

  // Print "<label> <emoji>" to Serial
  Serial.print(label);
  Serial.print(" ");
  Serial.println(emoji);

  // Delay 200 ms
  delay(200);
}