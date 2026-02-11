/*
  Color-based Object Classifier
  Board: Arduino Nano 33 BLE Sense
  Sensors: APDS-9960 RGB Color Sensor
  Inference: TensorFlow Lite for Microcontrollers
  Output: Predicted class over Serial with emoji
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>

#include "model.h"  // provides: const unsigned char model[] = { ... }

// TensorFlow Lite Micro headers
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// -----------------------------------------------------------------------------
// Application configuration
// -----------------------------------------------------------------------------
static const unsigned long kSamplePeriodMs = 200;     // sampling period
static const int           kAveragingSamples = 4;     // average over N samples
static const size_t        kTensorArenaSize = 8192;   // per spec

// Labels and emoji map
static const char* kLabels[3] = { "Apple", "Banana", "Orange" };
static const char* kEmojis[3] = { "🍎", "🍌", "🍊" };

// -----------------------------------------------------------------------------
// TFLM globals
// -----------------------------------------------------------------------------
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static tflite::ErrorReporter* error_reporter = nullptr;
static tflite::MicroErrorReporter micro_error_reporter;

static const tflite::Model* tfl_model = nullptr;   // NOTE: different name than 'model' from model.h
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interpreter = nullptr;

static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static int argmaxFloat(const float* data, int n) {
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

static int argmaxUInt8(const uint8_t* data, int n) {
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

// Read kAveragingSamples readings from APDS, average them, and return normalized
// chromaticity (R,G,B) such that R+G+B = 1 (approximately), each in [0,1].
static bool readAveragedNormalizedRGB(float& r_out, float& g_out, float& b_out) {
  long sum_r = 0, sum_g = 0, sum_b = 0;
  int valid_samples = 0;

  for (int i = 0; i < kAveragingSamples; ++i) {
    // Wait briefly for a color sample to be available
    unsigned long t0 = millis();
    while (!APDS.colorAvailable() && (millis() - t0) < 20) {
      delay(1);
    }

    int r = 0, g = 0, b = 0;
    if (APDS.readColor(r, g, b)) {
      sum_r += r;
      sum_g += g;
      sum_b += b;
      valid_samples++;
    }

    delay(kSamplePeriodMs / kAveragingSamples);
  }

  if (valid_samples == 0) {
    return false;
  }

  const float avg_r = static_cast<float>(sum_r) / valid_samples;
  const float avg_g = static_cast<float>(sum_g) / valid_samples;
  const float avg_b = static_cast<float>(sum_b) / valid_samples;

  const float total = avg_r + avg_g + avg_b;
  if (total <= 0.0f) {
    return false;
  }

  r_out = clampf(avg_r / total, 0.0f, 1.0f);
  g_out = clampf(avg_g / total, 0.0f, 1.0f);
  b_out = clampf(avg_b / total, 0.0f, 1.0f);
  return true;
}

// Copy 3-channel RGB data into the input tensor, supporting float32 or uint8 input.
static bool copyInputRGB(float r, float g, float b) {
  if (!input) return false;

  // Support both flat [3] or [1,3] shapes, only first 3 elements are used.
  const int needed = 3;

  if (input->type == kTfLiteFloat32) {
    float* in = input->data.f;
    in[0] = r;
    in[1] = g;
    in[2] = b;
    return true;
  } else if (input->type == kTfLiteUInt8) {
    // Quantize: q = zero_point + real/scale
    const float scale = input->params.scale;
    const int32_t zp = input->params.zero_point;
    if (scale <= 0.0f) return false;

    uint8_t* in = input->data.uint8;
    auto q = [&](float v) -> uint8_t {
      int32_t qv = static_cast<int32_t>(roundf(v / scale)) + zp;
      if (qv < 0) qv = 0;
      if (qv > 255) qv = 255;
      return static_cast<uint8_t>(qv);
    };
    in[0] = q(r);
    in[1] = q(g);
    in[2] = q(b);
    return true;
  }

  return false;
}

// Postprocess: pick top class index from output tensor
static int getTopClassIndex() {
  if (!output) return -1;

  // Assume output has 3 elements
  if (output->type == kTfLiteFloat32) {
    const float* out = output->data.f;
    return argmaxFloat(out, 3);
  } else if (output->type == kTfLiteUInt8) {
    const uint8_t* out = output->data.uint8;
    return argmaxUInt8(out, 3);
  }
  return -1;
}

// -----------------------------------------------------------------------------
// Arduino setup/loop
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(5); }

  Serial.println("Color-based Object Classifier (APDS-9960 + TFLM)");
  Serial.println("Initializing sensor...");

  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS-9960 sensor.");
    while (true) { delay(1000); }
  }
  // Allow sensor to stabilize
  delay(200);

  Serial.println("Initializing TensorFlow Lite Micro...");

  error_reporter = &micro_error_reporter;

  // Load model from model.h (byte array named 'model')
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version mismatch. Model: ");
    Serial.print(tfl_model->version());
    Serial.print("  Expected: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (true) { delay(1000); }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Initialization complete. Starting inference...");
}

void loop() {
  static unsigned long last_ms = 0;
  const unsigned long now = millis();
  if (now - last_ms < kSamplePeriodMs) {
    delay(5);
    return;
  }
  last_ms = now;

  float r, g, b;
  if (!readAveragedNormalizedRGB(r, g, b)) {
    Serial.println("WARN: Failed to read RGB. Retrying...");
    return;
  }

  if (!copyInputRGB(r, g, b)) {
    Serial.println("ERROR: Failed to copy input to tensor.");
    delay(1000);
    return;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(1000);
    return;
  }

  const int top = getTopClassIndex();
  if (top < 0 || top >= 3) {
    Serial.println("ERROR: Invalid classification result.");
    return;
  }

  // Output result with emoji and normalized RGB used
  Serial.print("Predicted: ");
  Serial.print(kLabels[top]);
  Serial.print(" ");
  Serial.print(kEmojis[top]);
  Serial.print(" | RGB(norm): ");
  Serial.print(r, 3); Serial.print(", ");
  Serial.print(g, 3); Serial.print(", ");
  Serial.println(b, 3);
}