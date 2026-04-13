#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

// IMPORTANT: Base TFLM header must come before dependent headers
#include "TensorFlowLite.h"
#include "model.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

namespace {
// TFLite Micro globals (kept in namespace to avoid name collisions)
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflm_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena for TFLM
constexpr int kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Classification labels and emojis
const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// Inference pacing
unsigned long last_inference_ms = 0;
const unsigned long kInferenceIntervalMs = 500;
} // namespace

// Helper: Normalize RGB to chromaticity (r,g,b) such that r+g+b=1
// Returns false if sum is zero (invalid reading)
bool rgbToChromaticity(int r, int g, int b, float& fr, float& fg, float& fb) {
  long sum = static_cast<long>(r) + static_cast<long>(g) + static_cast<long>(b);
  if (sum <= 0) {
    return false;
  }
  fr = static_cast<float>(r) / static_cast<float>(sum);
  fg = static_cast<float>(g) / static_cast<float>(sum);
  fb = static_cast<float>(b) / static_cast<float>(sum);
  return true;
}

// Helper: Argmax for uint8 scores
int argmax_u8(const uint8_t* data, int len) {
  int idx = 0;
  uint8_t best = data[0];
  for (int i = 1; i < len; i++) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  // Initialize sensor I2C and APDS9960 color sensor
  if (!APDS.begin()) {
    Serial.println("ERR: Failed to initialize APDS9960 color sensor.");
    while (true) { delay(1000); }
  }
  // Optional: allow sensor to stabilize
  delay(200);

  // Phase 1.2: Declare ErrorReporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the model from model.h
  tflm_model = tflite::GetModel(model);
#ifdef TFLITE_SCHEMA_VERSION
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERR: Model schema mismatch. Model schema: ");
    Serial.print(tflm_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }
#endif

  // Phase 1.5: Resolve operators (use AllOpsResolver as fallback)
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
    tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERR: AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // Phase 1.8: Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Verify input tensor shape and type
  bool input_ok = true;
  if (input->type != kTfLiteFloat32) {
    Serial.println("ERR: Input tensor is not float32.");
    input_ok = false;
  }
  if (input->dims->size != 2) {
    Serial.println("ERR: Input tensor rank is not 2.");
    input_ok = false;
  } else {
    int dim0 = input->dims->data[0];
    int dim1 = input->dims->data[1];
    if (!(dim0 == 1 && dim1 == 3)) {
      Serial.print("ERR: Input dims are [");
      Serial.print(dim0);
      Serial.print(", ");
      Serial.print(dim1);
      Serial.println("], expected [1, 3].");
      input_ok = false;
    }
  }

  // Verify output tensor shape and type
  bool output_ok = true;
  if (output->type != kTfLiteUInt8) {
    Serial.println("ERR: Output tensor is not uint8.");
    output_ok = false;
  }
  if (output->dims->size != 2) {
    Serial.println("ERR: Output tensor rank is not 2.");
    output_ok = false;
  } else {
    int odim0 = output->dims->data[0];
    int odim1 = output->dims->data[1];
    if (!(odim0 == 1 && odim1 == 3)) {
      Serial.print("ERR: Output dims are [");
      Serial.print(odim0);
      Serial.print(", ");
      Serial.print(odim1);
      Serial.println("], expected [1, 3].");
      output_ok = false;
    }
  }

  if (!(input_ok && output_ok)) {
    Serial.println("ERR: Tensor shape/type validation failed. Halting.");
    while (true) { delay(1000); }
  }

  // Phase 1.9: Other components (e.g., BLE) can be initialized here if needed
  // BLE.begin(); // Not used in this application

  Serial.println("Object Classifier by Color is ready.");
}

void loop() {
  // Phase 2.1: Sensor Setup/Acquire
  if (millis() - last_inference_ms < kInferenceIntervalMs) {
    delay(5);
    return;
  }

  int r = 0, g = 0, b = 0;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b);
  } else {
    // No color reading available yet
    delay(5);
    return;
  }

  // Phase 2.2: Preprocessing (normalize to chromaticity)
  float fr = 0, fg = 0, fb = 0;
  if (!rgbToChromaticity(r, g, b, fr, fg, fb)) {
    // Invalid reading; skip this cycle
    return;
  }

  // Phase 3.1: Data Copy to input tensor
  input->data.f[0] = fr; // Red
  input->data.f[1] = fg; // Green
  input->data.f[2] = fb; // Blue

  // Phase 3.2: Invoke Interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERR: Inference failed.");
    return;
  }

  // Phase 4.1: Process Output (uint8 scores)
  const uint8_t* scores_u8 = output->data.uint8;
  int top_idx = argmax_u8(scores_u8, 3);

  // Optional: convert to probabilities for display
  float prob0 = scores_u8[0] / 255.0f;
  float prob1 = scores_u8[1] / 255.0f;
  float prob2 = scores_u8[2] / 255.0f;

  // Phase 4.2: Execute Application Behavior (print result with emoji)
  Serial.print("RGB raw: ");
  Serial.print(r); Serial.print(", ");
  Serial.print(g); Serial.print(", ");
  Serial.print(b); Serial.print(" | norm: ");
  Serial.print(fr, 3); Serial.print(", ");
  Serial.print(fg, 3); Serial.print(", ");
  Serial.print(fb, 3); Serial.print(" -> ");

  Serial.print("Class: ");
  Serial.print(kClassNames[top_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[top_idx]);
  Serial.print(" | scores: [");
  Serial.print(prob0, 2); Serial.print(", ");
  Serial.print(prob1, 2); Serial.print(", ");
  Serial.print(prob2, 2); Serial.println("]");

  last_inference_ms = millis();
}