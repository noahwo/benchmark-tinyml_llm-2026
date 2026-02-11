/*
  Object Classifier by Color
  Board: Arduino Nano 33 BLE Sense
  Sensors: APDS-9960 (RGB)
  Inference: TensorFlow Lite for Microcontrollers
  Notes:
    - Input: float32 RGB, normalized so R+G+B = 1
    - Output: 3-class scores, argmax -> ["Apple", "Banana", "Orange"]
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model.h"  // Provides: const unsigned char model[] = {...}

// Application configuration
static const uint32_t kBaudRate = 9600;
static const uint32_t kInferenceIntervalMs = 200;
static const float kConfidenceThreshold = 0.5f;
static const char* kUnknownLabel = "Unknown";

// Labels and emojis (UTF-8)
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// TFLM globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // 12 KB Tensor Arena as specified
  constexpr int kTensorArenaSize = 12288;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  // Use AllOpsResolver for maximum compatibility
  tflite::AllOpsResolver resolver;
  tflite::MicroErrorReporter micro_error_reporter;
}

// Utility: normalize raw RGB to unit-sum fractions
static void normalizeRGB(int r, int g, int b, float& nr, float& ng, float& nb) {
  float fr = static_cast<float>(r);
  float fg = static_cast<float>(g);
  float fb = static_cast<float>(b);
  float sum = fr + fg + fb;
  if (sum <= 0.0f) {
    nr = ng = nb = 0.0f;  // Degenerate; sensor read failed
  } else {
    nr = fr / sum;
    ng = fg / sum;
    nb = fb / sum;
  }
}

// Utility: dequantize a single value from a quantized tensor
static float dequantizeVal(int32_t q, float scale, int32_t zero_point) {
  return scale * (static_cast<float>(q) - static_cast<float>(zero_point));
}

// Compute argmax and (approx) confidence
// - For float32 outputs: apply softmax for a probability estimate
// - For quantized outputs (uint8/int8): dequantize and then softmax
static void argmaxWithConfidence(const TfLiteTensor* out, int& best_idx, float& best_prob) {
  best_idx = -1;
  best_prob = 0.0f;

  const int dims = out->dims->size;
  // Assume last dimension is classes
  const int classes = out->dims->data[dims - 1];

  // Temporary buffer for float logits
  float logits[8];  // Enough for 3 classes; adjust if needed
  int max_supported = sizeof(logits) / sizeof(logits[0]);
  int n = classes;
  if (n > max_supported) n = max_supported;  // Safety cap

  if (out->type == kTfLiteFloat32) {
    for (int i = 0; i < n; ++i) {
      logits[i] = out->data.f[i];
    }
  } else if (out->type == kTfLiteUInt8) {
    for (int i = 0; i < n; ++i) {
      logits[i] = dequantizeVal(out->data.uint8[i], out->params.scale, out->params.zero_point);
    }
  } else if (out->type == kTfLiteInt8) {
    for (int i = 0; i < n; ++i) {
      logits[i] = dequantizeVal(out->data.int8[i], out->params.scale, out->params.zero_point);
    }
  } else {
    // Unsupported type; fall back to zero confidence
    best_idx = 0;
    best_prob = 0.0f;
    return;
  }

  // Softmax for probability estimate
  float max_logit = logits[0];
  for (int i = 1; i < n; ++i) {
    if (logits[i] > max_logit) max_logit = logits[i];
  }
  float sum_exp = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum_exp += expf(logits[i] - max_logit);
  }
  float probs[8];
  for (int i = 0; i < n; ++i) {
    probs[i] = expf(logits[i] - max_logit) / sum_exp;
  }

  // Argmax
  best_idx = 0;
  best_prob = probs[0];
  for (int i = 1; i < n; ++i) {
    if (probs[i] > best_prob) {
      best_prob = probs[i];
      best_idx = i;
    }
  }
}

void setup() {
  Serial.begin(kBaudRate);
  // Avoid blocking forever if no serial monitor attached
  uint32_t start_wait = millis();
  while (!Serial && (millis() - start_wait < 4000)) {
    delay(10);
  }

  Serial.println("Object Classifier by Color (Nano 33 BLE Sense)");
  Serial.println("Initializing...");

  // Error reporter
  error_reporter = &micro_error_reporter;

  // Load TFLite model from the byte array symbol 'model' provided by model.h
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch. Model version: ");
    Serial.print(tfl_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) delay(1000);
  }

  // Create interpreter with tensor arena
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Report input/output details
  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.print("Output type: ");
  Serial.println(output->type);

  // Initialize APDS-9960 color sensor
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS-9960!");
    while (true) delay(1000);
  } else {
    Serial.println("APDS-9960 initialized.");
  }

  Serial.println("Setup complete.");
}

void loop() {
  static uint32_t last_inference_ms = 0;
  uint32_t now = millis();
  if (now - last_inference_ms < kInferenceIntervalMs) {
    // Maintain ~5 Hz inference (200 ms)
    return;
  }
  last_inference_ms = now;

  // Wait (briefly) for new color data
  int attempts = 0;
  while (!APDS.colorAvailable() && attempts < 20) {
    delay(5);
    attempts++;
  }

  int r = 0, g = 0, b = 0;
  APDS.readColor(r, g, b);

  // Normalize
  float nr = 0.0f, ng = 0.0f, nb = 0.0f;
  normalizeRGB(r, g, b, nr, ng, nb);

  // Prepare model input
  if (input->type == kTfLiteFloat32) {
    // Expected shape [1,3]
    input->data.f[0] = nr;
    input->data.f[1] = ng;
    input->data.f[2] = nb;
  } else if (input->type == kTfLiteUInt8) {
    // Quantize from [0,1] to uint8 using tensor parameters
    const float s = input->params.scale;
    const int zp = input->params.zero_point;
    auto q = [&](float v) -> uint8_t {
      int32_t qv = static_cast<int32_t>(roundf(v / s) + zp);
      if (qv < 0) qv = 0;
      if (qv > 255) qv = 255;
      return static_cast<uint8_t>(qv);
    };
    input->data.uint8[0] = q(nr);
    input->data.uint8[1] = q(ng);
    input->data.uint8[2] = q(nb);
  } else if (input->type == kTfLiteInt8) {
    // Quantize to int8
    const float s = input->params.scale;
    const int zp = input->params.zero_point;
    auto q = [&](float v) -> int8_t {
      int32_t qv = static_cast<int32_t>(roundf(v / s) + zp);
      if (qv < -128) qv = -128;
      if (qv > 127) qv = 127;
      return static_cast<int8_t>(qv);
    };
    input->data.int8[0] = q(nr);
    input->data.int8[1] = q(ng);
    input->data.int8[2] = q(nb);
  } else {
    Serial.println("Unsupported input tensor type.");
    delay(50);
    return;
  }

  // Inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed.");
    delay(50);
    return;
  }

  // Postprocessing: argmax and confidence
  int best_idx = -1;
  float best_prob = 0.0f;
  argmaxWithConfidence(output, best_idx, best_prob);

  // Decide label
  const char* label = kUnknownLabel;
  const char* emoji = "❓";
  if (best_idx >= 0 && best_idx < 3 && best_prob >= kConfidenceThreshold) {
    label = kClassNames[best_idx];
    emoji = kClassEmojis[best_idx];
  }

  // Print result
  Serial.print("RGB norm = [");
  Serial.print(nr, 3); Serial.print(", ");
  Serial.print(ng, 3); Serial.print(", ");
  Serial.print(nb, 3); Serial.print("]  ->  ");
  Serial.print(label);
  Serial.print(" ");
  Serial.print(emoji);
  Serial.print("  (p=");
  Serial.print(best_prob, 2);
  Serial.println(")");
}