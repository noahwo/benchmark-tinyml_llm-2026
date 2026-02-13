/*
  Color-based Object Classifier
  Board: Arduino Nano 33 BLE Sense
  Sensor: APDS9960 RGB Color Sensor (Arduino_APDS9960)
  Inference: TensorFlow Lite for Microcontrollers (Arduino_TensorFlowLite)

  Notes:
  - The TFLite flatbuffer bytes are provided by model.h as: const unsigned char model[] = {...}
  - To avoid symbol conflicts, we never redeclare "model" as a tflite::Model*.
  - We parse the flatbuffer bytes with tflite::GetModel(model) and store the pointer in tfl_model.
*/

#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>

// TFLM headers
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// The compiled model bytes (symbol: model) must come from model.h
#include "model.h"

// ----------------------------- Configuration -----------------------------
static const unsigned long kInferenceIntervalMs = 200; // 5 Hz
static const int kNumClasses = 3;
static const char* kLabels[kNumClasses] = {"Apple", "Banana", "Orange"};
static const char* kEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};
static const uint32_t kSerialBaud = 9600;
static const size_t kTensorArenaSize = 16384;

// ----------------------------- Globals -----------------------------------
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroErrorReporter micro_error_reporter;

const tflite::Model* tfl_model = nullptr;

// Reserve op resolver entries exactly for used ops
tflite::MicroMutableOpResolver<3> micro_op_resolver;

alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
const TfLiteTensor* output = nullptr;

unsigned long last_inference_ms = 0;

// ----------------------------- Helpers -----------------------------------
static float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static int argmax_f(const float* vals, int n) {
  int idx = 0;
  float best = vals[0];
  for (int i = 1; i < n; ++i) {
    if (vals[i] > best) {
      best = vals[i];
      idx = i;
    }
  }
  return idx;
}

// Dequantize uint8 logits/probabilities to float using tensor params
static void dequantize_uint8_to_float(const uint8_t* src, float* dst, int n, float scale, int zero_point) {
  for (int i = 0; i < n; ++i) {
    dst[i] = (static_cast<int>(src[i]) - zero_point) * scale;
  }
}

// Print vector with given precision
static void print_vector(const char* name, const float* v, int n, int decimals) {
  Serial.print(name);
  Serial.print(": [");
  for (int i = 0; i < n; ++i) {
    Serial.print(v[i], decimals);
    if (i < n - 1) Serial.print(", ");
  }
  Serial.println("]");
}

// ----------------------------- Arduino Setup -----------------------------
void setup() {
  Serial.begin(kSerialBaud);
  // Wait a short time for serial connection (helps in some environments)
  unsigned long start_wait = millis();
  while (!Serial && (millis() - start_wait < 2000)) {}

  Serial.println("Color-based Object Classifier (TFLM) - Starting...");

  // Initialize I2C/Sensor
  Wire.begin();
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor.");
    // Continue anyway; user can reset later
  } else {
    Serial.println("APDS9960 initialized.");
  }

  // Set up error reporter
  error_reporter = &micro_error_reporter;

  // Map the model from the flatbuffer bytes included via model.h
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version mismatch. Model: ");
    Serial.print(tfl_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Register only the required operators
  // Ops required by the provided spec: FULLY_CONNECTED, RELU, SOFTMAX
  // Note: In many models, RELU can be fused in FULLY_CONNECTED; including it here for completeness.
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("ERROR: Could not add FullyConnected op.");
    while (true) { delay(1000); }
  }
  if (micro_op_resolver.AddRelu() != kTfLiteOk) {
    Serial.println("WARNING: Could not add Relu op. It may be fused in FullyConnected.");
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("ERROR: Could not add Softmax op.");
    while (true) { delay(1000); }
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input/output checks
  Serial.print("Input type: ");
  Serial.println(input->type == kTfLiteFloat32 ? "float32" :
                 (input->type == kTfLiteUInt8 ? "uint8" : "other"));
  Serial.print("Output type: ");
  Serial.println(output->type == kTfLiteFloat32 ? "float32" :
                 (output->type == kTfLiteUInt8 ? "uint8" : "other"));

  Serial.println("Setup complete.\n");
}

// ----------------------------- Arduino Loop ------------------------------
void loop() {
  // Throttle inference rate
  unsigned long now = millis();
  if (now - last_inference_ms < kInferenceIntervalMs) {
    delay(5);
    return;
  }
  last_inference_ms = now;

  // Wait for color data to be available
  if (!APDS.colorAvailable()) {
    // No data yet; try next loop
    return;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw)) {
    // Failed to read; skip this iteration
    return;
  }

  // Preprocessing: normalize to unit vector in RGB space
  // r = R/(R+G+B); g = G/(R+G+B); b = B/(R+G+B); clip to [0,1]
  float sum = static_cast<float>(r_raw + g_raw + b_raw);
  if (sum <= 0.0f) sum = 1.0f; // avoid division by zero

  float r_n = clamp01(static_cast<float>(r_raw) / sum);
  float g_n = clamp01(static_cast<float>(g_raw) / sum);
  float b_n = clamp01(static_cast<float>(b_raw) / sum);

  // Prepare model input
  if (input->type == kTfLiteFloat32) {
    // Expected input dims: [1,3]
    input->data.f[0] = r_n;
    input->data.f[1] = g_n;
    input->data.f[2] = b_n;
  } else if (input->type == kTfLiteUInt8) {
    // Quantized input; use tensor params
    const float scale = input->params.scale;
    const int zero_point = input->params.zero_point;
    // Quantize: q = z + f/scale
    auto quantize = [&](float f) -> uint8_t {
      const float q = static_cast<float>(zero_point) + (f / scale);
      int qi = static_cast<int>(roundf(q));
      if (qi < 0) qi = 0;
      if (qi > 255) qi = 255;
      return static_cast<uint8_t>(qi);
    };
    input->data.uint8[0] = quantize(r_n);
    input->data.uint8[1] = quantize(g_n);
    input->data.uint8[2] = quantize(b_n);
  } else {
    Serial.println("ERROR: Unsupported input tensor type.");
    delay(200);
    return;
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(50);
    return;
  }

  // Retrieve and interpret output
  float probs[kNumClasses] = {0, 0, 0};

  if (output->type == kTfLiteFloat32) {
    // Softmax probabilities
    for (int i = 0; i < kNumClasses; ++i) {
      probs[i] = output->data.f[i];
    }
  } else if (output->type == kTfLiteUInt8) {
    // Dequantize to float
    dequantize_uint8_to_float(output->data.uint8, probs, kNumClasses, output->params.scale, output->params.zero_point);

    // If these are logits (not guaranteed), try to renormalize to sum=1
    float sum_p = probs[0] + probs[1] + probs[2];
    if (sum_p > 0.0f && (sum_p < 0.95f || sum_p > 1.05f)) {
      // Renormalize to be safe
      for (int i = 0; i < kNumClasses; ++i) probs[i] /= sum_p;
    }
  } else {
    Serial.println("ERROR: Unsupported output tensor type.");
    delay(200);
    return;
  }

  int pred = argmax_f(probs, kNumClasses);
  float conf = probs[pred];

  // Serial output
  Serial.println("---- Inference ----");
  Serial.print("RGB raw: [");
  Serial.print(r_raw); Serial.print(", ");
  Serial.print(g_raw); Serial.print(", ");
  Serial.print(b_raw); Serial.println("]");

  float in_feats[3] = {r_n, g_n, b_n};
  print_vector("RGB normalized (r,g,b)", in_feats, 3, 3);

  print_vector("Probabilities (Apple, Banana, Orange)", probs, kNumClasses, 3);

  Serial.print("Prediction: ");
  Serial.print(kLabels[pred]);
  Serial.print(" ");
  Serial.print(kEmojis[pred]);
  Serial.print("  Confidence: ");
  Serial.println(conf, 3);

  Serial.println();

  // Maintain target inference interval
  // (We already throttle at loop start; extra delay to stabilize sensor sampling)
  delay(5);
}