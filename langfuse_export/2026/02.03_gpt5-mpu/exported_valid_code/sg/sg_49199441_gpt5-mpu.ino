#include <Arduino.h>
#include <TensorFlowLite.h>  // Base library must be included before TFLM headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <ArduinoBLE.h>
#include <math.h>

#include "model.h"

// Tensor arena configuration
constexpr int kTensorArenaSize = 16384;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// TFLM components
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflm_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Application configuration
static const char* kClassNames[] = { "Apple", "Banana", "Orange" };
static const char* kClassEmojis[] = { "🍎", "🍌", "🍊" };
static const uint32_t kBaudRate = 9600;

// Utility: clamp functions (avoid template to prevent conflicts)
static inline float clampf(float v, float lo, float hi) {
  return (v < lo) ? lo : (v > hi ? hi : v);
}
static inline int clampi(int v, int lo, int hi) {
  return (v < lo) ? lo : (v > hi ? hi : v);
}

bool initSensor() {
  if (!APDS.begin()) {
    Serial.println("APDS9960 init failed. Check wiring and power.");
    return false;
  }
  // No additional configuration required for basic RGB readings.
  return true;
}

bool readAndNormalizeRGB(float features[3]) {
  static uint32_t lastReadMs = 0;
  // Poll sensor
  if (!APDS.colorAvailable()) {
    // Throttle polling
    if (millis() - lastReadMs < 5) {
      return false;
    }
    lastReadMs = millis();
    return false;
  }

  int r = 0, g = 0, b = 0;
  if (!APDS.readColor(r, g, b)) {
    return false;
  }

  // Normalize to sum = 1.0 to match dataset scale (approximately 0.16..0.64 per channel)
  float rf = static_cast<float>(r);
  float gf = static_cast<float>(g);
  float bf = static_cast<float>(b);
  float sum = rf + gf + bf;
  if (sum <= 0.0f || !isfinite(sum)) {
    return false;
  }

  features[0] = clampf(rf / sum, 0.0f, 1.0f);  // Red
  features[1] = clampf(gf / sum, 0.0f, 1.0f);  // Green
  features[2] = clampf(bf / sum, 0.0f, 1.0f);  // Blue
  return true;
}

int argmax_float(const float* vals, int n) {
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

int argmax_u8(const uint8_t* vals, int n) {
  int idx = 0;
  uint8_t best = vals[0];
  for (int i = 1; i < n; ++i) {
    if (vals[i] > best) {
      best = vals[i];
      idx = i;
    }
  }
  return idx;
}

void setup() {
  Serial.begin(kBaudRate);
  while (!Serial) { delay(10); }

  Wire.begin();
  // BLE is included; not used in this sketch.

  // Initialize sensor
  if (!initSensor()) {
    // Continue even if sensor fails to allow debugging
  }

  // Set up TFLM error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model (model byte array is provided by model.h as `model`)
  tflm_model = tflite::GetModel(model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported version %d.",
                           tflm_model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Resolve operators
  static tflite::AllOpsResolver resolver;

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
  }

  // Fetch I/O tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic input validation
  if (input->dims->size != 2 || input->dims->data[0] != 1 || input->dims->data[1] != 3) {
    error_reporter->Report("Unexpected input tensor shape");
  }
  if (!(input->type == kTfLiteFloat32 || input->type == kTfLiteUInt8)) {
    error_reporter->Report("Unexpected input type");
  }

  Serial.println("Object Classifier by Color - Ready");
}

void loop() {
  float features[3];
  if (!readAndNormalizeRGB(features)) {
    // Save power briefly while waiting for new data
    delay(5);
    return;
  }

  // Copy input data
  if (input->type == kTfLiteFloat32) {
    input->data.f[0] = features[0];
    input->data.f[1] = features[1];
    input->data.f[2] = features[2];
  } else if (input->type == kTfLiteUInt8) {
    // Quantize if model expects uint8 input
    const float s = input->params.scale;
    const int zp = input->params.zero_point;
    if (s > 0.0f) {
      input->data.uint8[0] = (uint8_t)clampi((int)roundf(features[0] / s) + zp, 0, 255);
      input->data.uint8[1] = (uint8_t)clampi((int)roundf(features[1] / s) + zp, 0, 255);
      input->data.uint8[2] = (uint8_t)clampi((int)roundf(features[2] / s) + zp, 0, 255);
    } else {
      input->data.uint8[0] = (uint8_t)clampi((int)roundf(features[0] * 255.0f), 0, 255);
      input->data.uint8[1] = (uint8_t)clampi((int)roundf(features[1] * 255.0f), 0, 255);
      input->data.uint8[2] = (uint8_t)clampi((int)roundf(features[2] * 255.0f), 0, 255);
    }
  }

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Process output
  int predicted = -1;
  float scores_f[3] = {0, 0, 0};

  if (output->type == kTfLiteUInt8) {
    const uint8_t* out_u8 = output->data.uint8;
    predicted = argmax_u8(out_u8, 3);

    // Dequantize for reporting if scale available
    const float s = output->params.scale;
    const int zp = output->params.zero_point;
    if (s > 0.0f) {
      for (int i = 0; i < 3; ++i) {
        scores_f[i] = (float(out_u8[i]) - float(zp)) * s;
      }
    } else {
      for (int i = 0; i < 3; ++i) {
        scores_f[i] = out_u8[i] / 255.0f;
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    const float* out_f = output->data.f;
    predicted = argmax_float(out_f, 3);
    for (int i = 0; i < 3; ++i) {
      scores_f[i] = out_f[i];
    }
  } else {
    error_reporter->Report("Unexpected output type");
    return;
  }

  // Print result with emoji
  Serial.print("Input (R,G,B norm): ");
  Serial.print(features[0], 3); Serial.print(", ");
  Serial.print(features[1], 3); Serial.print(", ");
  Serial.print(features[2], 3); Serial.print(" -> ");

  if (predicted >= 0 && predicted < 3) {
    Serial.print("Pred: ");
    Serial.print(kClassNames[predicted]);
    Serial.print(" ");
    Serial.print(kClassEmojis[predicted]);
    Serial.print("  Scores: [");
    Serial.print(scores_f[0], 3); Serial.print(", ");
    Serial.print(scores_f[1], 3); Serial.print(", ");
    Serial.print(scores_f[2], 3); Serial.println("]");
  } else {
    Serial.println("Prediction error");
  }

  delay(100);
}