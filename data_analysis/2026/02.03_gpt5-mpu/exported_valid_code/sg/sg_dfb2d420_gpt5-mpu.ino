#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro - include base first, then dependent headers
#include "TensorFlowLite.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Include the compiled TFLite model
#include "model.h"

// TFLM globals
namespace {
  // Error reporter
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Model pointer (renamed to avoid conflict with model array from model.h)
  const tflite::Model* tflm_model = nullptr;

  // Interpreter and op resolver
  tflite::AllOpsResolver resolver;
  tflite::MicroInterpreter* interpreter = nullptr;

  // Tensors
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Tensor arena
  constexpr int kTensorArenaSize = 8192;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  // Class labels and emojis
  const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
  const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};
}

// Helper: normalize RGB to sum to 1.0 (match dataset feature scaling)
static inline void normalizeRGB(int r, int g, int b, float out[3]) {
  float sum = (float)r + (float)g + (float)b;
  if (sum <= 0.0f) {
    out[0] = out[1] = out[2] = 0.0f;
    return;
  }
  out[0] = (float)r / sum;
  out[1] = (float)g / sum;
  out[2] = (float)b / sum;
}

// Print tensor info (for debugging)
static void printTensorInfo(const TfLiteTensor* t, const char* name) {
  if (!t) return;
  Serial.print(name);
  Serial.print(" type=");
  Serial.print(t->type);
  Serial.print(" dims=[");
  if (t->dims && t->dims->size > 0) {
    for (int i = 0; i < t->dims->size; i++) {
      Serial.print(t->dims->data[i]);
      if (i < t->dims->size - 1) Serial.print(", ");
    }
  }
  Serial.println("]");
}

void setup() {
  // Serial init
  Serial.begin(9600);
  unsigned long start_wait = millis();
  while (!Serial && (millis() - start_wait < 4000)) { /* wait up to 4s */ }

  Serial.println("Object Classifier by Color - starting up");

  // Initialize color sensor
  if (!APDS.begin()) {
    Serial.println("Error: Failed to initialize APDS9960 color sensor.");
    // Continue; attempt to run model without sensor will be skipped in loop
  } else {
    Serial.println("APDS9960 initialized.");
  }

  // Load model (use ::model from model.h to avoid name conflict)
  tflm_model = tflite::GetModel(::model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch! Model schema: ");
    Serial.print(tflm_model->version());
    Serial.print(" != Runtime schema: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor
  if (input->type != kTfLiteFloat32) {
    Serial.print("Unexpected input type. Expected kTfLiteFloat32, got: ");
    Serial.println(input->type);
    while (true) { delay(1000); }
  }
  if (!(input->dims && input->dims->size == 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.print("Unexpected input shape. Expected [1,3], got: ");
    printTensorInfo(input, "input");
    while (true) { delay(1000); }
  }

  // Optionally print tensor info
  printTensorInfo(input, "Input");
  printTensorInfo(output, "Output");

  Serial.println("Setup complete. Reading colors and running inference...");
}

void loop() {
  // Check if color data is available
  if (APDS.colorAvailable()) {
    int r = 0, g = 0, b = 0;
    // Read color
    bool ok = APDS.readColor(r, g, b);
    if (!ok) {
      // If read failed, try again soon
      delay(10);
      return;
    }

    // Preprocess: normalize to sum=1 to match dataset feature scaling
    float features[3];
    normalizeRGB(r, g, b, features);

    // Copy to input tensor
    input->data.f[0] = features[0]; // Red
    input->data.f[1] = features[1]; // Green
    input->data.f[2] = features[2]; // Blue

    // Invoke inference
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed!");
      delay(50);
      return;
    }

    // Postprocess: read output, handle quantized or float outputs
    int best_idx = 0;
    float scores[3] = {0, 0, 0};  // dequantized or float scores

    if (output->type == kTfLiteUInt8) {
      // Quantized uint8 output
      const uint8_t* out = output->data.uint8;
      const float scale = output->params.scale;
      const int zp = output->params.zero_point;

      // Dequantize to float scores if scale is sane; otherwise use raw
      for (int i = 0; i < 3; i++) {
        if (scale > 0.0f) {
          scores[i] = (static_cast<int>(out[i]) - zp) * scale;
        } else {
          scores[i] = static_cast<float>(out[i]);
        }
        if (scores[i] > scores[best_idx]) best_idx = i;
      }
    } else if (output->type == kTfLiteFloat32) {
      // Float output
      const float* out = output->data.f;
      for (int i = 0; i < 3; i++) {
        scores[i] = out[i];
        if (scores[i] > scores[best_idx]) best_idx = i;
      }
    } else {
      Serial.print("Unsupported output type: ");
      Serial.println(output->type);
      delay(50);
      return;
    }

    // Normalize scores to [0,1] for display (optional)
    float sum_scores = scores[0] + scores[1] + scores[2];
    float probs[3];
    if (sum_scores > 0.0f && isfinite(sum_scores)) {
      for (int i = 0; i < 3; i++) probs[i] = scores[i] / sum_scores;
    } else {
      // Fallback: use softmax-like normalization
      float maxv = max(scores[0], max(scores[1], scores[2]));
      float exps[3] = {expf(scores[0] - maxv), expf(scores[1] - maxv), expf(scores[2] - maxv)};
      float sumexp = exps[0] + exps[1] + exps[2];
      for (int i = 0; i < 3; i++) probs[i] = exps[i] / (sumexp > 0 ? sumexp : 1.0f);
    }

    // Output results
    Serial.print("RGB: ");
    Serial.print(r); Serial.print(", ");
    Serial.print(g); Serial.print(", ");
    Serial.print(b);

    Serial.print("  Norm: ");
    Serial.print(features[0], 3); Serial.print(", ");
    Serial.print(features[1], 3); Serial.print(", ");
    Serial.print(features[2], 3);

    Serial.print("  -> Pred: ");
    Serial.print(kClassNames[best_idx]);
    Serial.print(" ");
    Serial.print(kClassEmojis[best_idx]);

    Serial.print("  Probs: [");
    for (int i = 0; i < 3; i++) {
      Serial.print(probs[i], 3);
      if (i < 2) Serial.print(", ");
    }
    Serial.println("]");

    delay(200);
  } else {
    // No new color data yet
    delay(5);
  }
}