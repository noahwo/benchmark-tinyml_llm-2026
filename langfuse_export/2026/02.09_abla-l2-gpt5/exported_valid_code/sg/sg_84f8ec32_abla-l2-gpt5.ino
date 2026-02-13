#include <Arduino.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Compiled TFLite model array
#include "model.h"  // must define: const unsigned char model[] = {...};

namespace {

// ---- Application constants ----
constexpr int kBaudRate = 9600;
constexpr float kEps = 1e-6f;
constexpr uint32_t kSamplingPeriodMs = 100;  // 10 Hz
constexpr int kNumFeatures = 3;              // ["Red", "Green", "Blue"]
constexpr int kNumClasses = 3;               // ["Apple","Banana","Orange"]

// Class labels and matching emojis (index alignment required)
const char* kClassNames[kNumClasses] = {"Apple", "Banana", "Orange"};
const char* kClassEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};

// ---- TFLite Micro globals ----
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroErrorReporter micro_error_reporter;

// Use a different name than "model" to avoid collision with model[] from model.h
const tflite::Model* tflite_model = nullptr;

tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;

// Arena size per spec
constexpr int kTensorArenaSize = 20480;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Utility: compute flat element count of a tensor
int FlatSize(const TfLiteTensor* t) {
  int sz = 1;
  for (int i = 0; i < t->dims->size; ++i) {
    sz *= t->dims->data[i];
  }
  return sz;
}

}  // namespace

void setup() {
  // Serial init
  Serial.begin(kBaudRate);
  while (!Serial) {
    // wait for Serial on native USB boards
  }
  Serial.println("Color Object Classifier (RGB -> Emoji)");
  Serial.println("Board: Arduino Nano 33 BLE Sense");
  Serial.println("Initializing APDS9960 color sensor...");

  // Sensor init (halt on failure)
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor. Halting.");
    while (true) {
      delay(1000);
    }
  }
  Serial.println("APDS9960 initialized.");

  // TFLite Micro initialization
  error_reporter = &micro_error_reporter;

  // Load the TFLite model from the included array
  tflite_model = tflite::GetModel(model);
  if (tflite_model == nullptr) {
    Serial.println("ERROR: tflite_model is null. Halting.");
    while (true) delay(1000);
  }

  // Check schema version
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version mismatch. Model: ");
    Serial.print(tflite_model->version());
    Serial.print(" != Expected: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) delay(1000);
  }

  // Create interpreter (static lifetime to avoid heap fragmentation)
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed. Halting.");
    while (true) delay(1000);
  }

  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Guard checks: input tensor
  const int input_flat = FlatSize(input);
  if (input->type != kTfLiteFloat32) {
    Serial.println("ERROR: Input tensor is not float32.");
    while (true) delay(1000);
  }
  if (input_flat != kNumFeatures) {
    Serial.print("ERROR: Input tensor size mismatch. Got ");
    Serial.print(input_flat);
    Serial.print(", expected ");
    Serial.println(kNumFeatures);
    while (true) delay(1000);
  }

  // Guard checks: output tensor
  const int output_flat = FlatSize(output);
  if (output->type != kTfLiteUInt8) {
    Serial.println("ERROR: Output tensor is not uint8 (quantized).");
    while (true) delay(1000);
  }
  if (output_flat != kNumClasses) {
    Serial.print("ERROR: Output tensor size mismatch. Got ");
    Serial.print(output_flat);
    Serial.print(", expected ");
    Serial.println(kNumClasses);
    while (true) delay(1000);
  }

  // Validate class count matches output
  if (kNumClasses != output_flat) {
    Serial.println("ERROR: Number of labels does not match model output size.");
    while (true) delay(1000);
  }

  Serial.println("TFLite Micro initialized.");
  Serial.print("Classes: ");
  for (int i = 0; i < kNumClasses; ++i) {
    Serial.print(kClassNames[i]);
    Serial.print(" ");
  }
  Serial.println();
  Serial.println("Ready. Reading RGB and inferring at 10 Hz...");
}

void loop() {
  static uint32_t last_ms = 0;
  const uint32_t now = millis();
  if (now - last_ms < kSamplingPeriodMs) {
    // Maintain ~10 Hz
    delay(1);
    return;
  }
  last_ms = now;

  // Ensure a new color sample is available (non-blocking guard)
  if (!APDS.colorAvailable()) {
    // If not available yet, skip this cycle
    return;
  }

  // Read raw RGB
  int r_raw = 0, g_raw = 0, b_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw)) {
    // If read failed, skip this cycle
    return;
  }

  // Preprocessing: rgb_sum_normalization
  float rf = static_cast<float>(r_raw);
  float gf = static_cast<float>(g_raw);
  float bf = static_cast<float>(b_raw);
  float sum = rf + gf + bf;
  if (sum < kEps) {
    // Avoid divide-by-zero; provide neutral input
    input->data.f[0] = 0.0f;
    input->data.f[1] = 0.0f;
    input->data.f[2] = 0.0f;
  } else {
    input->data.f[0] = rf / sum;  // "Red"
    input->data.f[1] = gf / sum;  // "Green"
    input->data.f[2] = bf / sum;  // "Blue"
  }

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed.");
    return;
  }

  // Postprocessing: Argmax on quantized uint8 scores
  const uint8_t* scores = output->data.uint8;
  int best_idx = 0;
  uint8_t best_score = scores[0];
  for (int i = 1; i < kNumClasses; ++i) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
  }

  // Optional: convert to approximate confidence using known scale (0.00390625)
  // Confidence ~ score / 255.0
  float confidence = static_cast<float>(best_score) / 255.0f;

  // Output formatting: "Label Emoji | r g b (norm) | conf"
  Serial.print(kClassNames[best_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[best_idx]);
  Serial.print(" | rgb_norm = [");
  Serial.print(input->data.f[0], 3);
  Serial.print(", ");
  Serial.print(input->data.f[1], 3);
  Serial.print(", ");
  Serial.print(input->data.f[2], 3);
  Serial.print("]");
  Serial.print(" | conf ≈ ");
  Serial.println(confidence, 3);
}