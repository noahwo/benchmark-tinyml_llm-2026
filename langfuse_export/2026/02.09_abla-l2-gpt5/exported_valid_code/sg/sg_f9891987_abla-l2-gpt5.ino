/*
  Color Object Classifier
  - Board: Arduino Nano 33 BLE Sense
  - Sensor: APDS-9960 RGB (Arduino_APDS9960)
  - ML: TensorFlow Lite for Microcontrollers (Arduino_TensorFlowLite)
  - Model: included via "model.h" (byte array named `model`)
  - Input: [Red, Green, Blue] normalized so R+G+B=1 (float32 by spec)
  - Output: 3 classes (Apple, Banana, Orange) via argmax
  - Serial: 9600 baud, prints class and emoji
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

namespace {

// TFLM globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflm_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Arena
constexpr int kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// App constants
constexpr float kEpsilon = 1e-6f;
constexpr uint32_t kIntervalMs = 500;
constexpr int kNumFeatures = 3;  // R,G,B
constexpr int kNumClasses = 3;   // Apple, Banana, Orange

const char* kLabels[kNumClasses] = {"Apple", "Banana", "Orange"};
const char* kEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};

// Helpers
inline float clip01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

int argmax(const float* vals, int len) {
  int best_i = 0;
  float best_v = vals[0];
  for (int i = 1; i < len; ++i) {
    if (vals[i] > best_v) {
      best_v = vals[i];
      best_i = i;
    }
  }
  return best_i;
}

// Quantization helpers (only used if model I/O is quantized)
inline uint8_t quantize_u8(float x, float scale, int32_t zp) {
  int32_t q = static_cast<int32_t>(roundf(x / scale) + zp);
  if (q < 0) q = 0;
  if (q > 255) q = 255;
  return static_cast<uint8_t>(q);
}
inline int8_t quantize_i8(float x, float scale, int32_t zp) {
  int32_t q = static_cast<int32_t>(roundf(x / scale) + zp);
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return static_cast<int8_t>(q);
}

}  // namespace

void setup() {
  Serial.begin(9600);
  while (!Serial && millis() < 3000) { /* wait for Serial */ }

  // Sensor init
  if (!APDS.begin()) {
    Serial.println("APDS-9960 init failed! Check wiring.");
  } else {
    Serial.println("APDS-9960 ready.");
  }

  // TFLM setup
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load flatbuffer model from the byte array symbol `model` (from model.h)
  tflm_model = tflite::GetModel(model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(tflm_model->version());
    Serial.print(" not equal to runtime schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) delay(1000);
  }

  static tflite::AllOpsResolver resolver;

  // Create interpreter (static lifetime to avoid heap fragmentation)
  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input/output checks
  if (input == nullptr || output == nullptr) {
    Serial.println("Failed to get input/output tensor");
    while (true) delay(1000);
  }

  // Optional: print tensor types for debugging
  Serial.print("Input type: "); Serial.println(input->type);
  Serial.print("Output type: "); Serial.println(output->type);
  Serial.println("Setup complete.");
}

void loop() {
  // Read RGB from sensor
  int R = 0, G = 0, B = 0, A = 0;
  if (APDS.colorAvailable()) {
    APDS.readColor(R, G, B, A);
  } else {
    // If color not ready yet, wait a bit
    delay(10);
    return;
  }

  // Preprocessing: normalize to sum=1
  float sum = static_cast<float>(R + G + B);
  sum = (sum <= 0.0f) ? kEpsilon : sum;
  float r = clip01(static_cast<float>(R) / sum);
  float g = clip01(static_cast<float>(G) / sum);
  float b = clip01(static_cast<float>(B) / sum);

  // Copy features into input tensor
  if (input->type == kTfLiteFloat32) {
    float* in = input->data.f;
    in[0] = r;
    in[1] = g;
    in[2] = b;
  } else if (input->type == kTfLiteUInt8) {
    uint8_t* in = input->data.uint8;
    const float s = input->params.scale;
    const int32_t zp = input->params.zero_point;
    in[0] = quantize_u8(r, s, zp);
    in[1] = quantize_u8(g, s, zp);
    in[2] = quantize_u8(b, s, zp);
  } else if (input->type == kTfLiteInt8) {
    int8_t* in = input->data.int8;
    const float s = input->params.scale;
    const int32_t zp = input->params.zero_point;
    in[0] = quantize_i8(r, s, zp);
    in[1] = quantize_i8(g, s, zp);
    in[2] = quantize_i8(b, s, zp);
  } else {
    Serial.print("Unsupported input type: ");
    Serial.println(input->type);
    delay(kIntervalMs);
    return;
  }

  // Inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    delay(kIntervalMs);
    return;
  }

  // Postprocessing: get scores and argmax
  float scores[kNumClasses] = {0, 0, 0};
  if (output->type == kTfLiteFloat32) {
    float* out = output->data.f;
    for (int i = 0; i < kNumClasses; ++i) scores[i] = out[i];
  } else if (output->type == kTfLiteUInt8) {
    uint8_t* out = output->data.uint8;
    float s = output->params.scale;
    int32_t zp = output->params.zero_point;
    for (int i = 0; i < kNumClasses; ++i) {
      scores[i] = s * (static_cast<int32_t>(out[i]) - zp);
    }
  } else if (output->type == kTfLiteInt8) {
    int8_t* out = output->data.int8;
    float s = output->params.scale;
    int32_t zp = output->params.zero_point;
    for (int i = 0; i < kNumClasses; ++i) {
      scores[i] = s * (static_cast<int32_t>(out[i]) - zp);
    }
  } else {
    Serial.print("Unsupported output type: ");
    Serial.println(output->type);
    delay(kIntervalMs);
    return;
  }

  int pred = argmax(scores, kNumClasses);

  // Output over Serial
  Serial.print("RGB raw: ");
  Serial.print(R); Serial.print(", ");
  Serial.print(G); Serial.print(", ");
  Serial.print(B);
  Serial.print(" | norm: ");
  Serial.print(r, 3); Serial.print(", ");
  Serial.print(g, 3); Serial.print(", ");
  Serial.print(b, 3);
  Serial.print(" | Pred: ");
  Serial.print(kLabels[pred]);
  Serial.print(" ");
  Serial.println(kEmojis[pred]);

  delay(kIntervalMs);
}