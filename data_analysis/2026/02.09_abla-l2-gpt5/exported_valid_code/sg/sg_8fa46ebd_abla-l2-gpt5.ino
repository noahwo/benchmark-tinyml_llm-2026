/*
  Object Classifier by Color (Arduino Nano 33 BLE Sense)
  - Reads RGB from APDS9960
  - Normalizes to chromaticity (R,G,B sum to ~1)
  - Runs TensorFlow Lite for Microcontrollers model from model.h
  - Prints classification as emoji over Serial

  Followed GUIDELINE:
  1) Initialization: libraries, globals, arena, model, resolver, interpreter, tensors, sensors
  2) Preprocessing: normalize RGB
  3) Inference: copy -> invoke
  4) Postprocessing: argmax -> emoji output
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "model.h"  // Must define: const unsigned char model[] = { ... }

// TFLM headers
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Application constants/specs
static const uint32_t kBaudRate = 9600;
static const uint32_t kInferenceIntervalMs = 200;
static const size_t kNumClasses = 3;
static const char* kLabels[kNumClasses] = { "Apple", "Banana", "Orange" };
static const char* kEmojis[kNumClasses] = { "🍎", "🍌", "🍊" };

// Tensor arena (heap for TFLM)
constexpr int kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// TFLM globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflm_model = nullptr;              // avoid name clash with model[] from model.h
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Timing
static uint32_t last_inference_ms = 0;

// Utility: clamp float to [0,1]
static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

// Read and normalize RGB as chromaticity. Returns true if a new reading was processed.
bool readNormalizedRGB(float& r_norm, float& g_norm, float& b_norm) {
  if (!APDS.colorAvailable()) {
    return false;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  // Arduino_APDS9960 supports readColor(r, g, b)
  APDS.readColor(r_raw, g_raw, b_raw);

  // Sum and guard
  long sum = (long)r_raw + (long)g_raw + (long)b_raw;
  if (sum <= 0) {
    return false;
  }

  // Chromaticity normalization
  r_norm = clamp01((float)r_raw / (float)sum);
  g_norm = clamp01((float)g_raw / (float)sum);
  b_norm = clamp01((float)b_raw / (float)sum);
  return true;
}

// Convert an output logit/probability to float, supporting multiple output tensor types
float get_output_score(const TfLiteTensor* out, int index) {
  switch (out->type) {
    case kTfLiteFloat32:
      return out->data.f[index];
    case kTfLiteUInt8: {
      // Dequantize using scale/zero_point if provided
      const float scale = out->params.scale == 0 ? (1.0f / 255.0f) : out->params.scale;
      const int zp = out->params.zero_point;
      const int v = out->data.uint8[index];
      return (static_cast<int>(v) - zp) * scale;
    }
    case kTfLiteInt8: {
      const float scale = out->params.scale == 0 ? (1.0f / 128.0f) : out->params.scale;
      const int zp = out->params.zero_point;
      const int8_t v = out->data.int8[index];
      return (static_cast<int>(v) - zp) * scale;
    }
    default:
      return 0.0f;
  }
}

void setup() {
  // Serial
  Serial.begin(kBaudRate);
  while (!Serial) { delay(10); }

  // Sensor init
  if (!APDS.begin()) {
    Serial.println("APDS9960 init failed. Check wiring or board selection.");
    while (1) { delay(1000); }
  }
  // APDS color is enabled by default after begin()

  // Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model from model.h (byte array named 'model')
  tflm_model = tflite::GetModel(model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(tflm_model->version());
    Serial.print(" not equal to supported version ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  // Resolver (all ops to avoid op-missing issues)
  static tflite::AllOpsResolver resolver;

  // Interpreter
  static tflite::MicroInterpreter static_interpreter(
    tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  // Cache input/output tensor pointers
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input check (expects [1,3] float)
  if (input->type != kTfLiteFloat32 || input->dims->size < 2 || input->dims->data[input->dims->size - 1] != 3) {
    Serial.println("Warning: model input is not [1,3] float32 as expected.");
  }

  Serial.println("Color classifier ready. Showing emoji results at ~5 Hz.");
}

void loop() {
  const uint32_t now = millis();
  if (now - last_inference_ms < kInferenceIntervalMs) {
    // Poll sensor in background while waiting
    APDS.colorAvailable(); // no-op to keep I2C bus warm
    delay(5);
    return;
  }
  last_inference_ms = now;

  float r, g, b;
  if (!readNormalizedRGB(r, g, b)) {
    // No fresh data; try again next tick
    return;
  }

  // Inference: copy to input tensor (order: Red, Green, Blue)
  if (input->type == kTfLiteFloat32) {
    // Accepts [1,3] or [3] flattened buffer
    input->data.f[0] = r;
    input->data.f[1] = g;
    input->data.f[2] = b;
  } else if (input->type == kTfLiteUInt8) {
    // Quantize to uint8 using input scale/zero_point
    float scale = input->params.scale == 0 ? (1.0f / 255.0f) : input->params.scale;
    int zp = input->params.zero_point;
    auto q = [&](float x) -> uint8_t {
      int32_t v = static_cast<int32_t>(roundf(x / scale) + zp);
      if (v < 0) v = 0; if (v > 255) v = 255;
      return static_cast<uint8_t>(v);
    };
    input->data.uint8[0] = q(r);
    input->data.uint8[1] = q(g);
    input->data.uint8[2] = q(b);
  } else if (input->type == kTfLiteInt8) {
    // Quantize to int8 using input scale/zero_point
    float scale = input->params.scale == 0 ? (1.0f / 128.0f) : input->params.scale;
    int zp = input->params.zero_point;
    auto q = [&](float x) -> int8_t {
      int32_t v = static_cast<int32_t>(roundf(x / scale) + zp);
      if (v < -128) v = -128; if (v > 127) v = 127;
      return static_cast<int8_t>(v);
    };
    input->data.int8[0] = q(r);
    input->data.int8[1] = q(g);
    input->data.int8[2] = q(b);
  } else {
    Serial.println("Unsupported input tensor type.");
    return;
  }

  // Invoke
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed.");
    return;
  }

  // Postprocessing: argmax
  float best_score = -1e9f;
  int best_idx = 0;
  for (int i = 0; i < (int)kNumClasses; ++i) {
    const float s = get_output_score(output, i);
    if (s > best_score) {
      best_score = s;
      best_idx = i;
    }
  }

  // Emit emoji + label (UTF-8 over Serial)
  Serial.print(kEmojis[best_idx]);
  Serial.print(" ");
  Serial.print(kLabels[best_idx]);
  Serial.print("  RGBn=(");
  Serial.print(r, 3); Serial.print(", ");
  Serial.print(g, 3); Serial.print(", ");
  Serial.print(b, 3); Serial.print(")  scores=[");

  for (int i = 0; i < (int)kNumClasses; ++i) {
    Serial.print(get_output_score(output, i), 3);
    if (i < (int)kNumClasses - 1) Serial.print(", ");
  }
  Serial.println("]");
}