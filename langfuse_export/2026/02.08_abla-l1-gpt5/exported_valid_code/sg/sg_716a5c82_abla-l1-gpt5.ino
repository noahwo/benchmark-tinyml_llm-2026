#include <Arduino.h>
#include <TensorFlowLite.h>
#include <Arduino_APDS9960.h>
#include "model.h"

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Application constants
static const int kBaudRate = 9600;
static const uint32_t kInferenceIntervalMs = 200;
static const size_t kTensorArenaSize = 12288;

// Labels and Emojis (UTF-8)
static const char* kClassLabels[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// TFLM globals
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* tfl_model = nullptr;
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter* interpreter = nullptr;

  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
}

// Utility: Halt with message
static void Halt(const char* msg) {
  Serial.println(msg);
  while (true) {
    delay(1000);
  }
}

// Utility: Count elements in tensor dims
static int ElementCount(const TfLiteIntArray* dims) {
  int count = 1;
  for (int i = 0; i < dims->size; ++i) {
    count *= dims->data[i];
  }
  return count;
}

// Utility: Clamp integer to [0, 255]
static inline uint8_t clamp255(int v) {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

void setup() {
  // Serial init
  Serial.begin(kBaudRate);
  while (!Serial) {
    delay(10);
  }

  // Initialize APDS-9960 Color Sensor
  if (!APDS.begin()) {
    Halt("ERROR: Failed to initialize APDS-9960 sensor.");
  }

  // Initialize TFLite Micro
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema ");
    Serial.print(tfl_model->version());
    Serial.print(" not equal to supported schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    Halt("Halting.");
  }

  // Create interpreter
  interpreter = new tflite::MicroInterpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Halt("ERROR: AllocateTensors() failed.");
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Verify input tensor type and shape
  if (input->type != kTfLiteFloat32) {
    Halt("ERROR: Input tensor type is not float32.");
  }
  if (ElementCount(input->dims) != 3) {
    Halt("ERROR: Input tensor does not have exactly 3 elements.");
  }

  // Verify output tensor type and shape
  if (output->type != kTfLiteUInt8) {
    Halt("ERROR: Output tensor type is not uint8.");
  }
  if (ElementCount(output->dims) != 3) {
    Halt("ERROR: Output tensor does not have exactly 3 elements.");
  }

  // Ready banner
  Serial.println("=======================================");
  Serial.println("RGB Object Classifier (TensorFlow Lite)");
  Serial.println("Device: Arduino Nano 33 BLE Sense");
  Serial.println("Sensor: APDS-9960 RGB Color Sensor (I2C 0x39)");
  Serial.println("Classes: Apple 🍎, Banana 🍌, Orange 🍊");
  Serial.println("Input: [Red, Green, Blue] -> float32 normalized [0..1]");
  Serial.println("Output: 3x uint8 scores -> argmax");
  Serial.println("Baud: 9600, Interval: 200 ms");
  Serial.println("Ready.");
  Serial.println("=======================================");
}

void loop() {
  // Wait for color data to be available
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0, c_raw = 0;
  APDS.readColor(r_raw, g_raw, b_raw, c_raw);

  // Preprocessing: clamp to [0, 255], convert to float32, normalize to [0, 1]
  uint8_t r8 = clamp255(r_raw);
  uint8_t g8 = clamp255(g_raw);
  uint8_t b8 = clamp255(b_raw);

  input->data.f[0] = static_cast<float>(r8) / 255.0f;
  input->data.f[1] = static_cast<float>(g8) / 255.0f;
  input->data.f[2] = static_cast<float>(b8) / 255.0f;

  // Inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(kInferenceIntervalMs);
    return;
  }

  // Postprocessing: read 3 uint8 scores and compute argmax
  const uint8_t s0 = output->data.uint8[0];
  const uint8_t s1 = output->data.uint8[1];
  const uint8_t s2 = output->data.uint8[2];

  int max_index = 0;
  uint8_t max_val = s0;
  if (s1 > max_val) { max_val = s1; max_index = 1; }
  if (s2 > max_val) { max_val = s2; max_index = 2; }

  // Map index to class label and emoji
  const char* label = kClassLabels[max_index];
  const char* emoji = kClassEmojis[max_index];

  // Serial output (UTF-8)
  Serial.print("Class: ");
  Serial.print(label);
  Serial.print(" ");
  Serial.println(emoji);

  delay(kInferenceIntervalMs);
}