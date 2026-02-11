/*
  Project: Color Object Classifier (RGB ➜ Emoji)
  Board:   Arduino Nano 33 BLE Sense
  Desc:    Uses onboard APDS-9960 RGB sensor + TensorFlow Lite Micro model to classify
           objects (Apple, Banana, Orange). Prints result with emojis over Serial.

  Note:
  - Model binary is provided via model.h as: const unsigned char model[] = {...};
  - Avoid naming conflicts: we use tflite_model for the parsed model pointer.
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "model.h"  // Must define: const unsigned char model[];

// ------------------------- Application Constants -------------------------
static const uint32_t kBaudRate = 9600;
static const uint32_t kLoopIntervalMs = 100;  // ~10 Hz
static const int kNumChannels = 3;
static const char* kLabels[kNumChannels] = {"Apple", "Banana", "Orange"};
static const char* kEmojis[kNumChannels] = {"🍎", "🍌", "🍊"};

// ------------------------- TFLM Globals ----------------------------------
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
tflite::AllOpsResolver resolver;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena (adjust size if needed).
constexpr int kTensorArenaSize = 16384;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// ------------------------- Utility Functions -----------------------------
static inline float clamp01f(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

static inline uint8_t clip_u8(int32_t v) {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

static inline int8_t clip_i8(int32_t v) {
  if (v < -128) return -128;
  if (v > 127) return 127;
  return static_cast<int8_t>(v);
}

static void writeInputRGBChromaticity(TfLiteTensor* in, float r, float g, float b) {
  // Input shape expected: [1, 3]
  // Handle float or quantized input gracefully.
  if (in->type == kTfLiteFloat32) {
    in->data.f[0] = r;
    in->data.f[1] = g;
    in->data.f[2] = b;
  } else if (in->type == kTfLiteUInt8) {
    // q = z + r / s
    const float inv_s = (in->params.scale == 0.f) ? 0.f : (1.0f / in->params.scale);
    in->data.uint8[0] = clip_u8(lroundf(in->params.zero_point + r * inv_s));
    in->data.uint8[1] = clip_u8(lroundf(in->params.zero_point + g * inv_s));
    in->data.uint8[2] = clip_u8(lroundf(in->params.zero_point + b * inv_s));
  } else if (in->type == kTfLiteInt8) {
    const float inv_s = (in->params.scale == 0.f) ? 0.f : (1.0f / in->params.scale);
    in->data.int8[0] = clip_i8(lroundf(in->params.zero_point + r * inv_s));
    in->data.int8[1] = clip_i8(lroundf(in->params.zero_point + g * inv_s));
    in->data.int8[2] = clip_i8(lroundf(in->params.zero_point + b * inv_s));
  } else {
    // Unsupported type; zero out as a fallback.
    for (int i = 0; i < 3; ++i) {
      if (in->type == kTfLiteInt16) {
        in->data.i16[i] = 0;
      } else if (in->type == kTfLiteInt32) {
        in->data.i32[i] = 0;
      }
    }
  }
}

static float getOutputValueAt(const TfLiteTensor* out, int i) {
  switch (out->type) {
    case kTfLiteFloat32:
      return out->data.f[i];
    case kTfLiteUInt8:
      // Dequantize for readability; argmax would be same on raw.
      return (static_cast<int32_t>(out->data.uint8[i]) - out->params.zero_point) * out->params.scale;
    case kTfLiteInt8:
      return (static_cast<int32_t>(out->data.int8[i]) - out->params.zero_point) * out->params.scale;
    default:
      return 0.0f;
  }
}

static int argmaxOutput(const TfLiteTensor* out) {
  int n = 1;
  if (out->dims && out->dims->size > 0) {
    n = out->dims->data[out->dims->size - 1];
  } else {
    n = kNumChannels;  // Fallback
  }
  int best_i = 0;
  float best_v = getOutputValueAt(out, 0);
  for (int i = 1; i < n; ++i) {
    float v = getOutputValueAt(out, i);
    if (v > best_v) {
      best_v = v;
      best_i = i;
    }
  }
  return best_i;
}

// ------------------------- Arduino Setup/Loop ----------------------------
void setup() {
  Serial.begin(kBaudRate);
  while (!Serial) { delay(10); }

  // Sensor setup
  if (!APDS.begin()) {
    Serial.println("APDS-9960 init failed. Check wiring/board.");
    // Continue anyway to allow model initialization.
  } else {
    Serial.println("APDS-9960 initialized.");
  }

  // TFLM setup according to guideline:
  // - Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // - Load the model (do NOT name this variable 'model' to avoid conflict with model.h)
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(tflite_model->version());
    Serial.print(" not equal to runtime schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // - Resolve operators (AllOps for simplicity/robustness)
  //   resolver already constructed as global.

  // - Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // - Allocate memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // - Define model inputs/outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.print("Output type: ");
  Serial.println(output->type);
  Serial.println("Setup complete.");
}

void loop() {
  static uint32_t last_ms = 0;
  uint32_t now = millis();
  if (now - last_ms < kLoopIntervalMs) {
    // Maintain ~10 Hz loop rate.
    return;
  }
  last_ms = now;

  // Non-blocking: proceed only when a new color sample is available.
  if (!APDS.colorAvailable()) {
    // No new data yet; keep loop responsive.
    return;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw)) {
    // Read failed. Skip this cycle.
    return;
  }

  // Preprocessing: chromaticity normalization
  float fr = static_cast<float>(r_raw);
  float fg = static_cast<float>(g_raw);
  float fb = static_cast<float>(b_raw);
  float sum = fr + fg + fb;
  if (sum < 1.0f) sum = 1.0f;  // avoid divide-by-zero
  float rn = clamp01f(fr / sum);
  float gn = clamp01f(fg / sum);
  float bn = clamp01f(fb / sum);

  // Data copy to input tensor
  writeInputRGBChromaticity(input, rn, gn, bn);

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Postprocessing: decode class by argmax
  int idx = argmaxOutput(output);
  if (idx < 0 || idx >= kNumChannels) {
    Serial.println("Invalid classification index");
    return;
  }

  // Output
  Serial.print("rgb_norm=[");
  Serial.print(rn, 3); Serial.print(", ");
  Serial.print(gn, 3); Serial.print(", ");
  Serial.print(bn, 3); Serial.print("] -> ");
  Serial.print(kLabels[idx]);
  Serial.print(" ");
  Serial.println(kEmojis[idx]);
}