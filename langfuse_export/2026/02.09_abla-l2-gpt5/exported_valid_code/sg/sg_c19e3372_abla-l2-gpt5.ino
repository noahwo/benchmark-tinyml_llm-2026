#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"  // Provides: const unsigned char model[] = {...}

// ------------------------------
// Configuration and Globals
// ------------------------------
static const int kSerialBaud = 9600;
static const uint32_t kLoopDelayMs = 150;
static const float kEps = 1e-6f;

// Tensor arena size from spec
static const int kTensorArenaSize = 16384;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// TensorFlow Lite Micro globals
static tflite::ErrorReporter* error_reporter = nullptr;
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::AllOpsResolver resolver;
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// Labels and emojis
static const char* kLabels[3] = {"Apple", "Banana", "Orange"};
static const char* kEmojis[3] = {"🍎", "🍌", "🍊"};

// ------------------------------
// Utilities
// ------------------------------
static inline int argmax_u8(const uint8_t* data, int n) {
  int best_i = 0;
  uint8_t best_v = data[0];
  for (int i = 1; i < n; ++i) {
    if (data[i] > best_v) { best_v = data[i]; best_i = i; }
  }
  return best_i;
}

static inline int argmax_f32(const float* data, int n) {
  int best_i = 0;
  float best_v = data[0];
  for (int i = 1; i < n; ++i) {
    if (data[i] > best_v) { best_v = data[i]; best_i = i; }
  }
  return best_i;
}

// ------------------------------
// Arduino Setup
// ------------------------------
void setup() {
  Serial.begin(kSerialBaud);
  while (!Serial) { /* wait for serial */ }

  Serial.println("Color-based Object Classifier starting...");

  // Sensor init
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS-9960.");
  } else {
    Serial.println("APDS-9960 initialized.");
  }

  // TF Lite Micro init
  error_reporter = &micro_error_reporter;

  // Load model from C array named 'model' provided by model.h
  g_model = tflite::GetModel(model);
  if (g_model == nullptr) {
    Serial.println("ERROR: GetModel returned null.");
    while (1) delay(1000);
  }
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema ");
    Serial.print(g_model->version());
    Serial.print(" not equal to supported schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) delay(1000);
  }

  // Create interpreter with tensor arena
  static tflite::MicroInterpreter static_interpreter(
      g_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (1) delay(1000);
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input/output checks
  if (input->type != kTfLiteFloat32 || input->dims->size < 2 || input->dims->data[0] != 1 || input->dims->data[1] != 3) {
    Serial.println("WARNING: Unexpected input tensor shape/type. Expected [1,3] float32.");
  }
  if (!(output->type == kTfLiteUInt8 || output->type == kTfLiteFloat32)) {
    Serial.println("WARNING: Unexpected output tensor type. Expected uint8 or float32.");
  }

  Serial.println("Setup complete.");
}

// ------------------------------
// Arduino Loop
// ------------------------------
void loop() {
  // Wait for a new color sample
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  // Read raw RGB
  int r_raw = 0, g_raw = 0, b_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw)) {
    // If read fails, try again
    delay(5);
    return;
  }

  // Preprocessing: clamp negatives and normalize to sum=1
  if (r_raw < 0) r_raw = 0;
  if (g_raw < 0) g_raw = 0;
  if (b_raw < 0) b_raw = 0;
  const float sum = (float)r_raw + (float)g_raw + (float)b_raw + kEps;
  const float r_n = (float)r_raw / sum;
  const float g_n = (float)g_raw / sum;
  const float b_n = (float)b_raw / sum;

  // Copy to input tensor [Red, Green, Blue]
  if (input->type == kTfLiteFloat32) {
    input->data.f[0] = r_n;
    input->data.f[1] = g_n;
    input->data.f[2] = b_n;
  } else {
    // Fallback for unexpected input quantization: simple clamp/scale to uint8
    // This path is not expected per spec (float32 input), but kept for robustness.
    input->data.uint8[0] = (uint8_t)(max(0.0f, min(1.0f, r_n)) * 255.0f + 0.5f);
    input->data.uint8[1] = (uint8_t)(max(0.0f, min(1.0f, g_n)) * 255.0f + 0.5f);
    input->data.uint8[2] = (uint8_t)(max(0.0f, min(1.0f, b_n)) * 255.0f + 0.5f);
  }

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Invoke failed.");
    delay(kLoopDelayMs);
    return;
  }

  // Postprocessing: argmax over classes
  int idx = 0;
  if (output->type == kTfLiteUInt8) {
    idx = argmax_u8(output->data.uint8, 3);
  } else {
    idx = argmax_f32(output->data.f, 3);
  }

  // Safety clamp
  if (idx < 0) idx = 0;
  if (idx > 2) idx = 2;

  // Output
  Serial.print(kLabels[idx]);
  Serial.print(" ");
  Serial.println(kEmojis[idx]);

  delay(kLoopDelayMs);
}