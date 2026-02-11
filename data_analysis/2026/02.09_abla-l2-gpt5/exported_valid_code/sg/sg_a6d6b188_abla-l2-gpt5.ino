/*
  Object Classifier by Color (Arduino Nano 33 BLE Sense)
  Version: 1.0.1

  - Uses onboard APDS-9960 color sensor.
  - Classifies objects (Apple, Banana, Orange) using a TensorFlow Lite Micro model.
  - Prints label and emoji over Serial.

  Notes:
  - Input features: [Red, Green, Blue] normalized by Clear channel: x_n = clamp(x / (c + 1e-6), 0.0, 1.0)
  - Output classes (order): ["Apple", "Banana", "Orange"]
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro
#include <TensorFlowLite.h>
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Compiled TFLite model binary as a C array
#include "model.h"  // NOTE: provides: const unsigned char model[] = { ... }

// ====== Application constants ======
static const unsigned long kSerialBaud = 9600;
static const uint32_t kInferenceIntervalMs = 500;
static const uint32_t kSensorStabilizationDelayMs = 50;

// Labels and emojis as specified
static const char* kClassLabels[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// ====== TFLite Micro globals (kept in an anonymous namespace) ======
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroErrorReporter micro_error_reporter;

const tflite::Model* g_tflModel = nullptr;
tflite::AllOpsResolver resolver;

constexpr int kTensorArenaSize = 16384;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
}  // namespace

// ====== Helper utilities ======
static inline float clamp01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

static bool SetupTFLite() {
  error_reporter = &micro_error_reporter;

  // Map the model array (from model.h) to a usable model
  g_tflModel = tflite::GetModel(model);
  if (g_tflModel == nullptr) {
    if (error_reporter) error_reporter->Report("GetModel returned null.");
    return false;
  }

  if (g_tflModel->version() != TFLITE_SCHEMA_VERSION) {
    if (error_reporter) {
      error_reporter->Report(
        "Model schema %d not equal to supported version %d.",
        g_tflModel->version(), TFLITE_SCHEMA_VERSION
      );
    }
    return false;
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    g_tflModel, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    if (error_reporter) error_reporter->Report("AllocateTensors() failed");
    return false;
  }

  // Cache input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate expected I/O shapes and types
  // Input: [1, 3], float32
  if (!(input && input->type == kTfLiteFloat32 && input->dims && input->dims->size >= 2 &&
        input->dims->data[input->dims->size - 1] == 3)) {
    if (error_reporter) error_reporter->Report("Unexpected input tensor spec.");
    return false;
  }

  // Output: [1, 3], typically uint8 per spec; fallback if float32
  if (!(output && output->dims && output->dims->size >= 2 &&
        output->dims->data[output->dims->size - 1] == 3)) {
    if (error_reporter) error_reporter->Report("Unexpected output tensor spec.");
    return false;
  }

  return true;
}

static bool SetupSensor() {
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS-9960.");
    return false;
  }
  // Optional: additional configuration could be added here.
  return true;
}

static bool ReadNormalizedRGB(float& r_n, float& g_n, float& b_n) {
  // Ensure color data is available
  if (!APDS.colorAvailable()) {
    return false;
  }

  int r = 0, g = 0, b = 0, c = 0;
  if (!APDS.readColor(r, g, b, c)) {
    return false;
  }

  // Normalization by clear channel with clamp to [0,1]
  const float denom = (float)c + 1e-6f;
  r_n = clamp01((float)r / denom);
  g_n = clamp01((float)g / denom);
  b_n = clamp01((float)b / denom);

  return true;
}

static int ArgMax3(const float a0, const float a1, const float a2) {
  int idx = 0;
  float best = a0;
  if (a1 > best) { best = a1; idx = 1; }
  if (a2 > best) { idx = 2; }
  return idx;
}

static int ArgMax3u8(const uint8_t a0, const uint8_t a1, const uint8_t a2) {
  int idx = 0;
  uint8_t best = a0;
  if (a1 > best) { best = a1; idx = 1; }
  if (a2 > best) { idx = 2; }
  return idx;
}

// ====== Arduino setup/loop ======
void setup() {
  Serial.begin(kSerialBaud);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color (Nano 33 BLE Sense)");
  Serial.println("Initializing sensor and TFLite...");

  if (!SetupSensor()) {
    Serial.println("Sensor init failed. Halting.");
    while (true) { delay(1000); }
  }

  if (!SetupTFLite()) {
    Serial.println("TFLite init failed. Halting.");
    while (true) { delay(1000); }
  }

  Serial.println("Initialization complete.");
}

void loop() {
  static uint32_t last_inference_ms = 0;
  const uint32_t now = millis();
  if (now - last_inference_ms < kInferenceIntervalMs) {
    delay(1);
    return;
  }
  last_inference_ms = now;

  // Precheck: ensure a color sample is available, wait a short stabilization time
  if (!APDS.colorAvailable()) {
    // Try once more after a brief delay
    delay(kSensorStabilizationDelayMs);
    if (!APDS.colorAvailable()) {
      Serial.println("No color data available.");
      return;
    }
  }

  // Read and normalize RGB
  float r_n = 0.0f, g_n = 0.0f, b_n = 0.0f;
  if (!ReadNormalizedRGB(r_n, g_n, b_n)) {
    Serial.println("Failed to read color.");
    return;
  }

  // Data copy to input tensor in order: [Red, Green, Blue]
  float* input_data = input->data.f;
  input_data[0] = r_n;
  input_data[1] = g_n;
  input_data[2] = b_n;

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed.");
    return;
  }

  // Postprocessing: argmax over 3 classes with bounds check
  int class_idx = -1;
  if (output->type == kTfLiteUInt8) {
    const uint8_t* y = output->data.uint8;
    class_idx = ArgMax3u8(y[0], y[1], y[2]);
  } else if (output->type == kTfLiteFloat32) {
    const float* y = output->data.f;
    class_idx = ArgMax3(y[0], y[1], y[2]);
  } else {
    Serial.println("Unsupported output tensor type.");
    return;
  }

  if (class_idx < 0 || class_idx > 2) {
    Serial.println("Invalid class index.");
    return;
  }

  // Output result
  Serial.print("Input RGBn = [");
  Serial.print(r_n, 3); Serial.print(", ");
  Serial.print(g_n, 3); Serial.print(", ");
  Serial.print(b_n, 3); Serial.print("] => ");

  Serial.print(kClassLabels[class_idx]);
  Serial.print(" ");
  Serial.println(kClassEmojis[class_idx]);
}