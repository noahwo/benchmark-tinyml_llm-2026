/*
  Project: Color-based Object Classifier
  Board:   Arduino Nano 33 BLE Sense
  Sensor:  APDS9960 (RGB)
  Inference: TensorFlow Lite for Microcontrollers

  Behavior:
    - Reads RGB from APDS9960
    - Averages N samples, clips to [0,255], normalizes to [0,1]
    - Runs inference on a 3-feature TFLM model included via model.h
    - Prints top class with emoji over Serial

  Notes to maintainer:
    - Avoid naming collisions with the byte array symbol `model` from model.h.
      We use `g_tflm_model` for the parsed TFLM model pointer.
    - When reading color, pass lvalue ints to APDS.readColor(int&,int&,int&).
      Do NOT pass casts or different integer types to avoid binding errors.
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>

// TensorFlow Lite Micro
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Model bytes
#include "model.h"

// -------------------- Configuration --------------------
static const unsigned long kInferenceIntervalMs = 300;
static const uint8_t       kAveragingSamples    = 4;
static const int           kTensorArenaSize     = 10240;

static const char* kLabels[3] = { "Apple", "Banana", "Orange" };
static const char* kEmojis[3] = { "🍎", "🍌", "🍊" };

// -------------------- TFLM Globals ---------------------
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model*   g_tflm_model   = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input  = nullptr;
  TfLiteTensor* output = nullptr;

  // Only required ops
  tflite::MicroMutableOpResolver<4> micro_op_resolver;

  // Tensor arena
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// -------------------- Utilities ------------------------
static inline uint8_t clipToByte(int v) {
  if (v < 0)   return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

static inline float clipTo255Float(float v) {
  if (v < 0.0f)   return 0.0f;
  if (v > 255.0f) return 255.0f;
  return v;
}

// Average N RGB readings from APDS9960, return float averages in [0,255]
void readAveragedRGB(uint8_t samples, float& r_out, float& g_out, float& b_out) {
  long sum_r = 0;
  long sum_g = 0;
  long sum_b = 0;

  for (uint8_t i = 0; i < samples; ++i) {
    // Ensure data ready
    while (!APDS.colorAvailable()) {
      delay(5);
    }

    // Must be lvalue ints to match signature: bool readColor(int& r, int& g, int& b);
    int r = 0, g = 0, b = 0;
    APDS.readColor(r, g, b);

    sum_r += r;
    sum_g += g;
    sum_b += b;

    delay(5);
  }

  const float inv = 1.0f / static_cast<float>(samples);
  r_out = clipTo255Float(static_cast<float>(sum_r) * inv);
  g_out = clipTo255Float(static_cast<float>(sum_g) * inv);
  b_out = clipTo255Float(static_cast<float>(sum_b) * inv);
}

// Write normalized features into the input tensor
void writeModelInput(float r01, float g01, float b01) {
  if (input->type == kTfLiteFloat32) {
    input->data.f[0] = r01;
    input->data.f[1] = g01;
    input->data.f[2] = b01;
  } else if (input->type == kTfLiteUInt8) {
    // Quantize to uint8 using tensor params
    const float scale = input->params.scale;
    const int   zp    = input->params.zero_point;
    auto q = [&](float x) -> uint8_t {
      int32_t qv = static_cast<int32_t>(roundf(x / scale) + zp);
      if (qv < 0)   qv = 0;
      if (qv > 255) qv = 255;
      return static_cast<uint8_t>(qv);
    };
    input->data.uint8[0] = q(r01);
    input->data.uint8[1] = q(g01);
    input->data.uint8[2] = q(b01);
  } else if (input->type == kTfLiteInt8) {
    // Quantize to int8 using tensor params
    const float scale = input->params.scale;
    const int   zp    = input->params.zero_point;
    auto q = [&](float x) -> int8_t {
      int32_t qv = static_cast<int32_t>(roundf(x / scale) + zp);
      if (qv < -128) qv = -128;
      if (qv > 127)  qv = 127;
      return static_cast<int8_t>(qv);
    };
    input->data.int8[0] = q(r01);
    input->data.int8[1] = q(g01);
    input->data.int8[2] = q(b01);
  } else {
    Serial.println("Unsupported input tensor type.");
  }
}

// Read scores from output tensor and return index of max score
int getTopClassIndex() {
  int top_idx = 0;

  if (output->type == kTfLiteFloat32) {
    float best = output->data.f[0];
    for (int i = 1; i < output->bytes / sizeof(float); ++i) {
      float v = output->data.f[i];
      if (v > best) { best = v; top_idx = i; }
    }
  } else if (output->type == kTfLiteUInt8) {
    uint8_t best = output->data.uint8[0];
    for (int i = 1; i < output->bytes; ++i) {
      uint8_t v = output->data.uint8[i];
      if (v > best) { best = v; top_idx = i; }
    }
  } else if (output->type == kTfLiteInt8) {
    int8_t best = output->data.int8[0];
    for (int i = 1; i < output->bytes; ++i) {
      int8_t v = output->data.int8[i];
      if (v > best) { best = v; top_idx = i; }
    }
  } else {
    Serial.println("Unsupported output tensor type.");
  }

  return top_idx;
}

// -------------------- Arduino Setup/Loop ---------------
void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Color-based Object Classifier (TFLM) - Nano 33 BLE Sense");

  // Initialize APDS9960
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960.");
    while (1) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // TFLM Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Parse the TFLM model from the C array symbol `model` defined in model.h
  g_tflm_model = tflite::GetModel(::model);
  if (g_tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch. Found: ");
    Serial.print(g_tflm_model->version());
    Serial.print(" Expected: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  // Register only needed ops
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    g_tflm_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed.");
    while (1) { delay(1000); }
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Input type: ");  Serial.println(input->type);
  Serial.print("Output type: "); Serial.println(output->type);
  Serial.println("Setup complete.");
}

void loop() {
  static unsigned long last_ms = 0;
  const unsigned long now = millis();
  if (now - last_ms < kInferenceIntervalMs) return;
  last_ms = now;

  // 1) Read and average sensor
  float r, g, b;
  readAveragedRGB(kAveragingSamples, r, g, b);

  // 2) Normalize [0,255] -> [0,1]
  const float inv255 = 1.0f / 255.0f;
  float r01 = r * inv255;
  float g01 = g * inv255;
  float b01 = b * inv255;

  // 3) Populate model input
  writeModelInput(r01, g01, b01);

  // 4) Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed.");
    return;
  }

  // 5) Postprocess and print
  int top = getTopClassIndex();
  const char* label = kLabels[top];
  const char* emoji = kEmojis[top];

  Serial.print("RGB(avg) ");
  Serial.print("R:"); Serial.print((int)clipToByte((int)roundf(r)));
  Serial.print(" G:");  Serial.print((int)clipToByte((int)roundf(g)));
  Serial.print(" B:");  Serial.print((int)clipToByte((int)roundf(b)));
  Serial.print("  ->  ");
  Serial.print(label);
  Serial.print(" ");
  Serial.println(emoji);
}