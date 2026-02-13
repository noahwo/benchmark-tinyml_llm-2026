/*
  Color Object Classifier
  - Board: Arduino Nano 33 BLE Sense
  - Sensor: APDS-9960 (RGB color)
  - Inference: TensorFlow Lite for Microcontrollers
  - Output: Serial prints with Unicode emojis
  - Model input: float32 [1,3] in order ["Red","Green","Blue"], normalized to [0..1]
  - Model output: uint8 [1,3] with labels ["Apple","Banana","Orange"]

  NOTE:
  - This sketch avoids naming collisions with the byte array 'model' defined in model.h
    by using 'tflm_model' for the tflite::Model* handle.
*/

#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// The compiled TFLite model as a C array
#include "model.h"

// ------------------------------
// Globals (TensorFlow Lite Micro)
// ------------------------------
namespace {
  // Error reporter (prints to Serial when available)
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // The TFLite model structure reference
  const tflite::Model* tflm_model = nullptr;

  // Contains implementations of all the operations to run the model
  tflite::AllOpsResolver resolver;

  // Interpreter that will run the model
  tflite::MicroInterpreter* interpreter = nullptr;

  // Model input and output tensors
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena (must be large enough for the model’s tensors)
  constexpr int kTensorArenaSize = 20480;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];

  // Application labels and emojis
  constexpr int kNumClasses = 3;
  const char* kLabels[kNumClasses] = { "Apple", "Banana", "Orange" };
  const char* kEmojis[kNumClasses] = { "🍎", "🍌", "🍊" };
}

// ------------------------------
// Utility Functions
// ------------------------------
static inline float clip01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static void printModelIOInfo() {
  if (!input || !output) return;

  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.print("Input dims: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print("x");
  }
  Serial.println();

  Serial.print("Output type: ");
  Serial.println(output->type);
  Serial.print("Output dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print("x");
  }
  Serial.println();
}

// ------------------------------
// Arduino Setup
// ------------------------------
void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); } // Wait for Serial on native USB boards

  Serial.println("Color Object Classifier (APDS-9960 + TFLM)");

  // Initialize sensor
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS-9960 sensor.");
    while (1) { delay(100); }
  }
  Serial.println("APDS-9960 initialized.");

  // Load TFLite model (note: 'model' is defined in model.h)
  tflm_model = tflite::GetModel(::model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema mismatch. Model schema: ");
    Serial.print(tflm_model->version());
    Serial.print(" != Supported: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(100); }
  }

  // Create an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (1) { delay(100); }
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TFLM initialized. Model IO:");
  printModelIOInfo();

  Serial.println("Setup complete. Reading colors and classifying...");
}

// ------------------------------
// Arduino Loop
// ------------------------------
void loop() {
  // Wait until a color sample is available
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  // Read raw RGBC values
  int r = 0, g = 0, b = 0, c = 0; // 'c' = clear/ambient
  // The library generally supports the 4-argument variant
  APDS.readColor(r, g, b, c);

  // Normalize to unit-sum RGB in [0..1]
  const float sum = static_cast<float>(r + g + b);
  float rf = 0.0f, gf = 0.0f, bf = 0.0f;
  if (sum > 0.0f) {
    rf = clip01(static_cast<float>(r) / sum);
    gf = clip01(static_cast<float>(g) / sum);
    bf = clip01(static_cast<float>(b) / sum);
  }
  // Optional: guard for pathological cases
  if (!isfinite(rf) || !isfinite(gf) || !isfinite(bf)) {
    rf = gf = bf = 0.0f;
  }

  // Copy data to model input according to its type
  if (input->type == kTfLiteFloat32) {
    // Expected input order: ["Red","Green","Blue"]
    input->data.f[0] = rf;
    input->data.f[1] = gf;
    input->data.f[2] = bf;
  } else if (input->type == kTfLiteUInt8) {
    // Quantize from float [0..1] to uint8 using tensor quantization params
    const float scale = input->params.scale;
    const int32_t zp = input->params.zero_point;
    input->data.uint8[0] = static_cast<uint8_t>(roundf(rf / scale) + zp);
    input->data.uint8[1] = static_cast<uint8_t>(roundf(gf / scale) + zp);
    input->data.uint8[2] = static_cast<uint8_t>(roundf(bf / scale) + zp);
  } else if (input->type == kTfLiteInt8) {
    // Quantize to int8 if necessary
    const float scale = input->params.scale;
    const int32_t zp = input->params.zero_point;
    input->data.int8[0] = static_cast<int8_t>(roundf(rf / scale) + zp);
    input->data.int8[1] = static_cast<int8_t>(roundf(gf / scale) + zp);
    input->data.int8[2] = static_cast<int8_t>(roundf(bf / scale) + zp);
  } else {
    Serial.print("ERROR: Unsupported input tensor type: ");
    Serial.println(input->type);
    delay(250);
    return;
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(250);
    return;
  }

  // Read output and compute argmax
  int best_idx = 0;
  float best_score = -1e9f;

  if (output->type == kTfLiteUInt8) {
    for (int i = 0; i < kNumClasses; i++) {
      const float score = static_cast<float>(output->data.uint8[i]); // 0..255
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < kNumClasses; i++) {
      const float score = output->data.f[i];
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
  } else if (output->type == kTfLiteInt8) {
    // Dequantize to compare
    const float scale = output->params.scale;
    const int32_t zp = output->params.zero_point;
    for (int i = 0; i < kNumClasses; i++) {
      const float score = (static_cast<int>(output->data.int8[i]) - zp) * scale;
      if (score > best_score) {
        best_score = score;
        best_idx = i;
      }
    }
  } else {
    Serial.print("ERROR: Unsupported output tensor type: ");
    Serial.println(output->type);
    delay(250);
    return;
  }

  // Prepare a readable confidence metric based on output type
  float conf_pct = 0.0f;
  if (output->type == kTfLiteUInt8) {
    conf_pct = (best_score / 255.0f) * 100.0f;
  } else {
    // For float/int8, clamp to [0..1] if plausible, else just present raw as percentage
    float s = best_score;
    if (isfinite(s)) {
      if (output->type == kTfLiteFloat32) {
        if (s < 0.0f) s = 0.0f;
        if (s > 1.0f) s = 1.0f;
        conf_pct = s * 100.0f;
      } else { // int8 dequantized, may be any real number; just map heuristically
        // If model is logits, this won't be true prob. Keep as raw scaled percent for display.
        conf_pct = s * 100.0f;
      }
    }
  }

  // Print result
  Serial.print("RGB raw=(");
  Serial.print(r); Serial.print(", ");
  Serial.print(g); Serial.print(", ");
  Serial.print(b); Serial.print(")  norm=(");
  Serial.print(rf, 3); Serial.print(", ");
  Serial.print(gf, 3); Serial.print(", ");
  Serial.print(bf, 3); Serial.print(")  ->  ");
  Serial.print(kLabels[best_idx]);
  Serial.print(" ");
  Serial.print(kEmojis[best_idx]);
  Serial.print("  (");
  Serial.print(conf_pct, 1);
  Serial.println("%)");

  delay(200);
}