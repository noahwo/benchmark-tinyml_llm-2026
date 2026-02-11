#include <TensorFlowLite.h>                 // Base TFLM header (must come before dependent headers)
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino_APDS9960.h>
#include <Wire.h>
#include "model.h"                          // Contains the TFLite flatbuffer array (e.g., model[])

// TensorFlow Lite Micro globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;

  constexpr int kTensorArenaSize = 16 * 1024;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
  tflite::MicroErrorReporter micro_error_reporter;
}

static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

void failAndHalt(const char* msg) {
  Serial.println(msg);
  while (true) {
    delay(1000);
  }
}

void setup() {
  // Phase 1: Initialization
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color - TinyML (Nano 33 BLE Sense)");

  // Initialize sensor (APDS9960 RGB)
  if (!APDS.begin()) {
    failAndHalt("ERROR: APDS9960 initialization failed.");
  }
  Serial.println("APDS9960 initialized.");

  // TFLM setup
  error_reporter = &micro_error_reporter;

  // Load model from model.h
  tfl_model = tflite::GetModel(model); // 'model' is the flatbuffer array from model.h
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    failAndHalt("ERROR: Model schema version mismatch.");
  }

  // Resolve operators - use AllOpsResolver as a safe default
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensor buffers
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    failAndHalt("ERROR: AllocateTensors() failed.");
  }

  // Retrieve input tensor and validate
  input = interpreter->input(0);
  if (input == nullptr) {
    failAndHalt("ERROR: Failed to get input tensor.");
  }

  // Expect input: [1,3], float32
  bool dims_ok = (input->dims != nullptr) &&
                 (input->dims->size >= 2) &&
                 (input->dims->data[0] == 1) &&
                 (input->dims->data[1] == 3);
  if (!dims_ok || input->type != kTfLiteFloat32) {
    failAndHalt("ERROR: Unexpected input tensor shape or type (expect [1,3] float32).");
  }

  Serial.println("TFLM initialized. Ready to classify colors.");
}

void loop() {
  // Phase 2: Preprocessing - read sensor and normalize
  // Wait for new color data
  while (!APDS.colorAvailable()) {
    delay(5);
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  APDS.readColor(r_raw, g_raw, b_raw);

  // Normalize to chromaticity (sum to ~1) to match dataset style
  // Avoid guessing sensor absolute scale by using ratios.
  float sum = static_cast<float>(r_raw) + static_cast<float>(g_raw) + static_cast<float>(b_raw);
  float r_n = 0.0f, g_n = 0.0f, b_n = 0.0f;

  if (sum > 0.0f) {
    r_n = static_cast<float>(r_raw) / sum;
    g_n = static_cast<float>(g_raw) / sum;
    b_n = static_cast<float>(b_raw) / sum;
  } else {
    // If sensor returns zeros, skip this cycle
    delay(50);
    return;
  }

  // Phase 3: Inference
  // Copy normalized data into input tensor
  input->data.f[0] = r_n;
  input->data.f[1] = g_n;
  input->data.f[2] = b_n;

  // Invoke the model
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(100);
    return;
  }

  // Phase 4: Postprocessing - interpret model output
  TfLiteTensor* output = interpreter->output(0);
  if (output == nullptr) {
    Serial.println("ERROR: Output tensor is null.");
    delay(100);
    return;
  }
  if (output->type != kTfLiteUInt8) {
    Serial.println("ERROR: Unexpected output type (expect uint8).");
    delay(100);
    return;
  }

  // Expect output shape [1,3]
  int out_classes = 3;
  if (!(output->dims && output->dims->size >= 2 && output->dims->data[0] == 1 && output->dims->data[1] == out_classes)) {
    Serial.println("ERROR: Unexpected output tensor shape (expect [1,3]).");
    delay(100);
    return;
  }

  // Read scores and compute argmax
  uint8_t scores[3];
  for (int i = 0; i < out_classes; i++) {
    scores[i] = output->data.uint8[i];
  }

  int best_idx = 0;
  uint8_t best_score = scores[0];
  for (int i = 1; i < out_classes; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
  }

  // Optional: derive a pseudo-confidence from 0..255
  float confidence = best_score / 255.0f;

  // Execute application behavior: print result with emoji
  Serial.print("RGB(raw)=");
  Serial.print(r_raw); Serial.print(",");
  Serial.print(g_raw); Serial.print(",");
  Serial.print(b_raw);

  Serial.print("  RGB(norm)=");
  Serial.print(r_n, 3); Serial.print(",");
  Serial.print(g_n, 3); Serial.print(",");
  Serial.print(b_n, 3);

  Serial.print("  Scores=[");
  Serial.print(scores[0]); Serial.print(",");
  Serial.print(scores[1]); Serial.print(",");
  Serial.print(scores[2]); Serial.print("]  =>  ");

  Serial.print(kClassNames[best_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[best_idx]);
  Serial.print("  conf=");
  Serial.println(confidence, 2);

  delay(150);
}