#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>

// Base TFLite Micro header must come before dependent headers
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Model file (must be provided alongside the sketch)
#include "model.h"

// TensorFlow Lite Micro global objects and buffers
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tfl_model = nullptr;  // renamed to avoid conflict with array symbol 'model' from model.h
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena: adjust per model needs; specified as 20*1024
  constexpr int kTensorArenaSize = 20 * 1024;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  // Class names and emojis
  const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
  const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

  // Helper: argmax over N elements
  int argmax_u8(const uint8_t* data, int n) {
    int idx = 0;
    uint8_t best = data[0];
    for (int i = 1; i < n; ++i) {
      if (data[i] > best) {
        best = data[i];
        idx = i;
      }
    }
    return idx;
  }

  int argmax_f32(const float* data, int n) {
    int idx = 0;
    float best = data[0];
    for (int i = 1; i < n; ++i) {
      if (data[i] > best) {
        best = data[i];
        idx = i;
      }
    }
    return idx;
  }
}

void setup() {
  // Phase 1: Initialization
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color - TinyML");
  Serial.println("Initializing...");

  // Initialize sensor (APDS9960 RGB color)
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor.");
    while (1) { delay(100); }
  }
  Serial.println("APDS9960 initialized.");

  // Set up TFLite Micro error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model from model.h (expects a symbol like model[])
  tfl_model = tflite::GetModel(::model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema ");
    Serial.print(tfl_model->version());
    Serial.print(" not equal to supported schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(100); }
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (1) { delay(100); }
  }

  // Access model input and check compatibility
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Input checks
  bool input_ok = true;
  if (input->type != kTfLiteFloat32) {
    Serial.println("ERROR: Input tensor must be float32.");
    input_ok = false;
  }
  if (!(input->dims->size == 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.print("ERROR: Unexpected input tensor shape. Got [");
    for (int i = 0; i < input->dims->size; i++) {
      Serial.print(input->dims->data[i]);
      if (i < input->dims->size - 1) Serial.print(", ");
    }
    Serial.println("], expected [1, 3].");
    input_ok = false;
  }
  if (!input_ok) {
    while (1) { delay(100); }
  }

  // Output checks
  bool output_ok = true;
  if (!((output->type == kTfLiteUInt8) || (output->type == kTfLiteFloat32))) {
    Serial.println("ERROR: Output tensor must be uint8 or float32.");
    output_ok = false;
  }
  if (!(output->dims->size == 2 && output->dims->data[0] == 1 && output->dims->data[1] == 3)) {
    Serial.print("ERROR: Unexpected output tensor shape. Got [");
    for (int i = 0; i < output->dims->size; i++) {
      Serial.print(output->dims->data[i]);
      if (i < output->dims->size - 1) Serial.print(", ");
    }
    Serial.println("], expected [1, 3].");
    output_ok = false;
  }
  if (!output_ok) {
    while (1) { delay(100); }
  }

  Serial.println("Initialization complete. Starting inference loop...");
}

void loop() {
  // Phase 2: Preprocessing - acquire sensor data
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0, a_raw = 0;
  APDS.readColor(r_raw, g_raw, b_raw, a_raw);

  // Normalize to chromaticity (sum to ~1.0) to match dataset
  float sum = static_cast<float>(r_raw + g_raw + b_raw);
  if (sum <= 0.0f) {
    // No valid reading
    delay(10);
    return;
  }

  float red = static_cast<float>(r_raw) / sum;
  float green = static_cast<float>(g_raw) / sum;
  float blue = static_cast<float>(b_raw) / sum;

  // Optional clamp
  if (red < 0) red = 0; if (red > 1) red = 1;
  if (green < 0) green = 0; if (green > 1) green = 1;
  if (blue < 0) blue = 0; if (blue > 1) blue = 1;

  // Phase 3: Inference
  // 3.1 Data Copy
  input->data.f[0] = red;
  input->data.f[1] = green;
  input->data.f[2] = blue;

  // 3.2 Invoke
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(50);
    return;
  }

  // Phase 4: Postprocessing
  int predicted_idx = 0;
  float scores_f[3] = {0, 0, 0};

  if (output->type == kTfLiteUInt8) {
    const uint8_t* out_u8 = output->data.uint8;
    predicted_idx = argmax_u8(out_u8, 3);
    // Dequantize to float for display (optional)
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    for (int i = 0; i < 3; ++i) {
      scores_f[i] = scale * (static_cast<int>(out_u8[i]) - zero_point);
    }
  } else { // kTfLiteFloat32
    const float* out_f = output->data.f;
    predicted_idx = argmax_f32(out_f, 3);
    for (int i = 0; i < 3; ++i) {
      scores_f[i] = out_f[i];
    }
  }

  // Execute application behavior: print result with emoji
  Serial.print("RGB raw: ");
  Serial.print(r_raw); Serial.print(", ");
  Serial.print(g_raw); Serial.print(", ");
  Serial.print(b_raw); Serial.print(" | Norm: ");
  Serial.print(red, 3); Serial.print(", ");
  Serial.print(green, 3); Serial.print(", ");
  Serial.print(blue, 3); Serial.print(" | Scores: ");
  Serial.print(scores_f[0], 3); Serial.print(", ");
  Serial.print(scores_f[1], 3); Serial.print(", ");
  Serial.print(scores_f[2], 3); Serial.print(" | Prediction: ");
  Serial.print(kClassEmojis[predicted_idx]);
  Serial.print(" ");
  Serial.println(kClassNames[predicted_idx]);

  delay(100);
}