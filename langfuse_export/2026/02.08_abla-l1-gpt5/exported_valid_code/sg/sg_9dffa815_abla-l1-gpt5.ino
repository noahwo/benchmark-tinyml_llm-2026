/*
  Sketch: RGB_Object_Classifier.ino
  Device: Arduino Nano 33 BLE Sense
  Sensor: APDS9960 RGB color sensor (I2C 0x39)
  Libraries:
    - TensorFlow Lite for Microcontrollers
    - Arduino_APDS9960
  Model: model.h (TFLite model as C array)
  Features: Red, Green, Blue
  Classes: Apple, Banana, Orange
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

// App/device configuration
static const unsigned long kInvokeIntervalMs = 100; // 10 Hz
static const uint32_t kSerialBaud = 9600;

// Model/configuration specifics
static const int kNumFeatures = 3; // R, G, B
static const int kNumClasses = 3;  // Apple, Banana, Orange
static const float kNormalizeScale = 0.00392156862745098f; // 1/255.0f

// Tensor arena size (bytes)
static const int kTensorArenaSize = 20480;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// TFLM globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Labels and emojis
static const char* kLabels[kNumClasses] = { "Apple", "Banana", "Orange" };
static const char* kEmojis[kNumClasses] = { "🍎", "🍌", "🍊" };

// Timing
static unsigned long last_invoke_ms = 0;

// Utility: clamp to 0..255 and convert to float
static inline float clamp255_to_float_norm(int v) {
  if (v < 0) v = 0;
  if (v > 255) v = 255;
  return v * kNormalizeScale;
}
static inline uint8_t clamp_u8(int v) {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return (uint8_t)v;
}

// Quantization helpers
static inline uint8_t quantize_uint8_from_real(float real, const TfLiteTensor* t) {
  // real_value = scale * (q - zero_point) => q = real/scale + zero_point
  const float inv_scale = (t->params.scale == 0.f) ? 0.f : (1.0f / t->params.scale);
  int q = static_cast<int>(real * inv_scale + t->params.zero_point + 0.5f);
  if (q < 0) q = 0;
  if (q > 255) q = 255;
  return static_cast<uint8_t>(q);
}
static inline int8_t quantize_int8_from_real(float real, const TfLiteTensor* t) {
  const float inv_scale = (t->params.scale == 0.f) ? 0.f : (1.0f / t->params.scale);
  int q = static_cast<int>(real * inv_scale + t->params.zero_point + 0.5f);
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return static_cast<int8_t>(q);
}

// Argmax over 3 elements for different tensor types
static int argmax_float(const float* v, int n) {
  int idx = 0;
  float best = v[0];
  for (int i = 1; i < n; ++i) {
    if (v[i] > best) { best = v[i]; idx = i; }
  }
  return idx;
}
static int argmax_uint8(const uint8_t* v, int n) {
  int idx = 0;
  uint8_t best = v[0];
  for (int i = 1; i < n; ++i) {
    if (v[i] > best) { best = v[i]; idx = i; }
  }
  return idx;
}
static int argmax_int8(const int8_t* v, int n) {
  int idx = 0;
  int8_t best = v[0];
  for (int i = 1; i < n; ++i) {
    if (v[i] > best) { best = v[i]; idx = i; }
  }
  return idx;
}

void setup() {
  Serial.begin(kSerialBaud);
  while (!Serial) { delay(10); }

  Serial.println("RGB Object Classifier - TensorFlow Lite Micro");
  Serial.println("Initializing APDS9960...");
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor.");
    while (1) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // TFLM setup
  error_reporter = &micro_error_reporter;

  // The model must be a tflite flatbuffer embedded via model.h as a C array named 'model'
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported %d.", tfl_model->version(), TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input/output checks
  if (input->dims->size < 2 || input->dims->data[input->dims->size - 1] != kNumFeatures) {
    error_reporter->Report("Unexpected input tensor shape.");
  }
  if (output->dims->size < 2 || output->dims->data[output->dims->size - 1] != kNumClasses) {
    error_reporter->Report("Unexpected output tensor shape.");
  }

  Serial.println("Setup complete. Starting inference loop...");
}

void loop() {
  const unsigned long now = millis();
  if (now - last_invoke_ms < kInvokeIntervalMs) {
    delay(1);
    return;
  }
  last_invoke_ms = now;

  // Wait for new color data
  if (!APDS.colorAvailable()) {
    return;
  }

  int r = 0, g = 0, b = 0, a = 0;
  if (!APDS.readColor(r, g, b, a)) {
    // If read fails, skip this cycle
    return;
  }

  // Preprocess: clamp to [0,255], min-max normalize to [0,1]
  float rf = clamp255_to_float_norm(r);
  float gf = clamp255_to_float_norm(g);
  float bf = clamp255_to_float_norm(b);

  // Write to model input according to input type
  if (input->type == kTfLiteFloat32) {
    float* in = input->data.f;
    in[0] = rf;
    in[1] = gf;
    in[2] = bf;
  } else if (input->type == kTfLiteUInt8) {
    // Quantize from real [0..1] to uint8 using tensor quantization params
    uint8_t* in = input->data.uint8;
    in[0] = quantize_uint8_from_real(rf, input);
    in[1] = quantize_uint8_from_real(gf, input);
    in[2] = quantize_uint8_from_real(bf, input);
  } else if (input->type == kTfLiteInt8) {
    int8_t* in = input->data.int8;
    in[0] = quantize_int8_from_real(rf, input);
    in[1] = quantize_int8_from_real(gf, input);
    in[2] = quantize_int8_from_real(bf, input);
  } else {
    // Unsupported input type
    return;
  }

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    // Invocation failed; skip this cycle
    return;
  }

  // Postprocessing: argmax on output
  int idx = 0;
  if (output->type == kTfLiteFloat32) {
    idx = argmax_float(output->data.f, kNumClasses);
  } else if (output->type == kTfLiteUInt8) {
    idx = argmax_uint8(output->data.uint8, kNumClasses);
  } else if (output->type == kTfLiteInt8) {
    idx = argmax_int8(output->data.int8, kNumClasses);
  } else {
    // Unsupported output type
    return;
  }

  // Emit result in requested format: "{label} {emoji}\n"
  Serial.print(kLabels[idx]);
  Serial.print(" ");
  Serial.println(kEmojis[idx]);
}