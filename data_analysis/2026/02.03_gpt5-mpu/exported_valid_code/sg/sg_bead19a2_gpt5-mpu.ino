#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <Arduino_LSM9DS1.h>
#include <Arduino_HTS221.h>

#include <TensorFlowLite.h>  // Base TFLM header must be included before micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model.h"  // Model file (must be included as per instructions)

// TensorFlow Lite Micro globals (kept as globals to persist for the lifetime of the application)
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tflm_model = nullptr;  // renamed to avoid conflict with model[] from model.h
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Adjust if you run out of memory or need more
  constexpr int kTensorArenaSize = 8192;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];

  // Classification labels
  const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
  const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};
}

// Helper: Normalize raw RGB to sum-1 floats in [0,1]; returns false if invalid
bool readNormalizedRGB(float& r_f, float& g_f, float& b_f) {
  int r, g, b, c;
  // Wait for color data to be available
  if (!APDS.colorAvailable()) {
    return false;
  }
  APDS.readColor(r, g, b, c);

  // Convert to float and normalize by channel sum (dataset appears to be normalized this way)
  const float rf = static_cast<float>(r);
  const float gf = static_cast<float>(g);
  const float bf = static_cast<float>(b);
  const float sum = rf + gf + bf;

  if (sum <= 0.0f || !isfinite(sum)) {
    return false;
  }

  r_f = rf / sum;
  g_f = gf / sum;
  b_f = bf / sum;

  // Clamp to [0,1] just in case
  r_f = r_f < 0.f ? 0.f : (r_f > 1.f ? 1.f : r_f);
  g_f = g_f < 0.f ? 0.f : (g_f > 1.f ? 1.f : g_f);
  b_f = b_f < 0.f ? 0.f : (b_f > 1.f ? 1.f : b_f);

  return true;
}

void setup() {
  // Phase 1.1: Include libraries and initialize serial
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  // Phase 1.9: Initialize peripherals
  Wire.begin();

  if (!APDS.begin()) {
    Serial.println("APDS9960 init failed. Check wiring or power.");
    while (1) { delay(100); }
  }
  APDS.setGestureSensitivity(80);  // Not used, but ensures device is configured
  Serial.println("APDS9960 ready.");

  // Optional init for additional sensors (not used in this app)
  IMU.begin();
  HTS.begin();

  // Phase 1.2: Declare ErrorReporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the model
  // Note: model.h provides a flatbuffer array symbol named 'model'
  tflm_model = tflite::GetModel(::model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported version %d.",
                           tflm_model->version(), TFLITE_SCHEMA_VERSION);
    Serial.println("Model schema version mismatch.");
    while (1) { delay(100); }
  }

  // Phase 1.5: Resolve operators
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    Serial.println("AllocateTensors failed.");
    while (1) { delay(100); }
  }

  // Phase 1.8: Define model inputs and outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor: expected [1,3], float32
  bool input_ok = (input != nullptr) &&
                  (input->type == kTfLiteFloat32) &&
                  (input->dims->size == 2) &&
                  (input->dims->data[0] == 1) &&
                  (input->dims->data[1] == 3);
  if (!input_ok) {
    Serial.println("Unexpected input tensor shape/type. Expected [1,3] float32.");
    while (1) { delay(100); }
  }

  // Validate output tensor: expected [1,3], uint8
  bool output_ok = (output != nullptr) &&
                   (output->type == kTfLiteUInt8) &&
                   (output->dims->size == 2) &&
                   (output->dims->data[0] == 1) &&
                   (output->dims->data[1] == 3);
  if (!output_ok) {
    Serial.println("Unexpected output tensor shape/type. Expected [1,3] uint8.");
    while (1) { delay(100); }
  }

  Serial.println("TinyML Object Classifier by Color initialized.");
  Serial.println("Bring a colored object in front of the sensor.");
}

void loop() {
  // Phase 2.1 + 2.2: Sensor setup and preprocessing
  float r_n, g_n, b_n;
  if (!readNormalizedRGB(r_n, g_n, b_n)) {
    delay(10);
    return;
  }

  // Phase 3.1: Copy preprocessed data into input tensor
  // Input order must match training: ['Red', 'Green', 'Blue']
  input->data.f[0] = r_n;
  input->data.f[1] = g_n;
  input->data.f[2] = b_n;

  // Phase 3.2: Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed.");
    delay(100);
    return;
  }

  // Phase 4.1: Process output (uint8 logits/probabilities)
  uint8_t scores[3];
  scores[0] = output->data.uint8[0];
  scores[1] = output->data.uint8[1];
  scores[2] = output->data.uint8[2];

  // Argmax
  int max_idx = 0;
  uint8_t max_val = scores[0];
  for (int i = 1; i < 3; ++i) {
    if (scores[i] > max_val) {
      max_val = scores[i];
      max_idx = i;
    }
  }

  // Optional: convert to percentage if desired (assumes uint8 quantization over [0,255])
  float prob_pct = (static_cast<float>(max_val) / 255.0f) * 100.0f;

  // Phase 4.2: Execute application behavior (print result with emoji)
  Serial.print("RGB(norm): R=");
  Serial.print(r_n, 3);
  Serial.print(" G=");
  Serial.print(g_n, 3);
  Serial.print(" B=");
  Serial.print(b_n, 3);
  Serial.print(" | Scores[u8]: [");
  Serial.print(scores[0]);
  Serial.print(", ");
  Serial.print(scores[1]);
  Serial.print(", ");
  Serial.print(scores[2]);
  Serial.print("] => Class: ");
  Serial.print(kClassNames[max_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[max_idx]);
  Serial.print(" (");
  Serial.print(prob_pct, 1);
  Serial.println("%)");

  delay(120);
}