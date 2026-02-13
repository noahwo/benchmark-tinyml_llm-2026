/*
  Color-Based Object Classifier
  Board: Arduino Nano 33 BLE Sense
  Sensors: Onboard APDS9960 RGB
  Libraries: Arduino_APDS9960, Arduino_TensorFlowLite
  Model: TensorFlow Lite Micro (included via model.h)
  Output: Class label + Unicode emoji over Serial
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro includes
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// The compiled TFLM model array symbol is defined here as: const unsigned char model[]
#include "model.h"

// -------- Application constants --------
static const uint32_t kBaudRate = 9600;
static const uint32_t kSamplingIntervalMs = 200;
static const uint8_t kNumClasses = 3;
static const char* kClassLabels[kNumClasses] = { "Apple", "Banana", "Orange" };
static const char* kEmojiApple = "\xF0\x9F\x8D\x8E";   // 🍎
static const char* kEmojiBanana = "\xF0\x9F\x8D\x8C";  // 🍌
static const char* kEmojiOrange = "\xF0\x9F\x8D\x8A";  // 🍊
static const char* kEmojiUnknown = "\xE2\x9D\x93";     // ❓
static const float kConfidenceThreshold = 0.5f;        // For uint8 outputs, threshold ~ 128/255

// -------- TFLM arena size (bytes) --------
constexpr int kTensorArenaSize = 16384;
static uint8_t tensor_arena[kTensorArenaSize];

// -------- TFLM globals --------
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* g_model = nullptr;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Utility: clamp float to [0,1]
static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

// Map predicted class index to emoji
const char* classEmoji(uint8_t idx) {
  switch (idx) {
    case 0: return kEmojiApple;
    case 1: return kEmojiBanana;
    case 2: return kEmojiOrange;
    default: return kEmojiUnknown;
  }
}

void setup() {
  // Serial setup
  Serial.begin(kBaudRate);
  while (!Serial) { delay(10); }

  Serial.println("Color-Based Object Classifier (APDS9960 + TFLM)");
  Serial.flush();

  // Sensor setup
  if (!APDS.begin()) {
    Serial.println("APDS9960 init failed. Check board and library.");
    while (true) { delay(100); }
  }
  Serial.println("APDS9960 ready.");

  // Load model from the C array symbol 'model' defined in model.h
  g_model = tflite::GetModel(model);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(g_model->version());
    Serial.print(" does not match TFLite schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(100); }
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    g_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) { delay(100); }
  }

  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input/output checks
  if (input->type != kTfLiteFloat32) {
    Serial.println("Model input must be float32.");
    while (true) { delay(100); }
  }
  if (output->type != kTfLiteUInt8) {
    Serial.println("Model output must be uint8.");
    while (true) { delay(100); }
  }

  // Warmup inference
  for (int i = 0; i < 3; ++i) input->data.f[i] = 0.0f;
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Warmup Invoke() failed");
    while (true) { delay(100); }
  }

  Serial.println("Setup complete. Starting inference loop...");
}

void loop() {
  static uint32_t last_ms = 0;
  uint32_t now = millis();
  if (now - last_ms < kSamplingIntervalMs) return;
  last_ms = now;

  // Wait for color data available
  if (!APDS.colorAvailable()) {
    // No new data; try again next tick
    return;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  APDS.readColor(r_raw, g_raw, b_raw);

  // Normalize to [0,1]. APDS9960 typical raw range ~0..1024 (varies with lighting).
  const float kSensorMax = 1024.0f;
  float r = clamp01(static_cast<float>(r_raw) / kSensorMax);
  float g = clamp01(static_cast<float>(g_raw) / kSensorMax);
  float b = clamp01(static_cast<float>(b_raw) / kSensorMax);

  // Copy features into model input [Red, Green, Blue]
  input->data.f[0] = r;
  input->data.f[1] = g;
  input->data.f[2] = b;

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Read uint8 output logits/scores
  // Expected shape [1, 3] for classes: Apple, Banana, Orange
  const uint8_t s0 = output->data.uint8[0];
  const uint8_t s1 = output->data.uint8[1];
  const uint8_t s2 = output->data.uint8[2];

  // Convert to [0,1] for confidence display
  const float p0 = s0 / 255.0f;
  const float p1 = s1 / 255.0f;
  const float p2 = s2 / 255.0f;

  // Argmax
  uint8_t idx = 0;
  uint8_t maxv = s0;
  if (s1 > maxv) { maxv = s1; idx = 1; }
  if (s2 > maxv) { maxv = s2; idx = 2; }

  const float conf = maxv / 255.0f;
  const bool confident = conf >= kConfidenceThreshold;

  // Map to label and emoji
  const char* label = confident ? kClassLabels[idx] : "Unknown";
  const char* emoji = confident ? classEmoji(idx) : kEmojiUnknown;

  // Output result
  Serial.print("RGB raw: ");
  Serial.print(r_raw); Serial.print(", ");
  Serial.print(g_raw); Serial.print(", ");
  Serial.print(b_raw); Serial.print(" | norm: ");
  Serial.print(r, 3); Serial.print(", ");
  Serial.print(g, 3); Serial.print(", ");
  Serial.print(b, 3); Serial.print(" | scores: ");
  Serial.print(p0, 2); Serial.print(", ");
  Serial.print(p1, 2); Serial.print(", ");
  Serial.print(p2, 2); Serial.print(" | pred: ");
  Serial.print(label); Serial.print(" ");
  Serial.println(emoji);
}