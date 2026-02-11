/*
  Color Object Classifier (RGB -> Emoji)
  Board: Arduino Nano 33 BLE Sense
  Sensor: APDS-9960 (onboard)
  Model: TensorFlow Lite Micro (included via model.h)

  - Reads RGB from APDS-9960
  - Preprocess: average 5 samples, normalize by sum (RGB ratios)
  - Inference interval: 500 ms
  - Output: Predicted label + emoji over Serial
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Compiled TFLM model array
#include "model.h"

// ---------- Application configuration ----------
static const uint32_t kBaudRate = 9600;
static const uint16_t kSamplesPerRead = 5;
static const uint16_t kInterSampleDelayMs = 10;
static const uint16_t kInferenceIntervalMs = 500;
static const uint8_t kWarmupInferences = 1;
static const int kNumFeatures = 3; // ['Red','Green','Blue']
static const int kNumClasses = 3;  // ['Apple','Banana','Orange']

// Class labels and emojis
static const char* kLabels[kNumClasses] = {"Apple", "Banana", "Orange"};
static const char* kEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};

// ---------- TFLM Globals ----------
namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* g_tflite_model = nullptr;
tflite::AllOpsResolver g_resolver;

constexpr int kTensorArenaSize = 12288;
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

// Last valid normalized sample (for zero-sum handling)
float g_last_valid_rgb[kNumFeatures] = {0.0f, 0.0f, 0.0f};
bool g_has_last_valid = false;
}  // namespace

// ---------- Utilities ----------
static inline float clip01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static uint8_t quantize_uint8(float x, float scale, int zero_point) {
  // General quantization: q = x/scale + zp, clamped to [0,255]
  if (scale <= 0.0f) {
    // Fallback: assume x in [0,1]
    int q = (int)lroundf(clip01(x) * 255.0f);
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return (uint8_t)q;
  }
  int q = (int)lroundf((x / scale) + (float)zero_point);
  if (q < 0) q = 0;
  if (q > 255) q = 255;
  return (uint8_t)q;
}

// Read and average kSamplesPerRead samples from APDS-9960
// Returns true on success, fills avg_r, avg_g, avg_b with averaged raw values.
static bool readAveragedRawRGB(float& avg_r, float& avg_g, float& avg_b) {
  uint32_t sum_r = 0, sum_g = 0, sum_b = 0;
  uint16_t valid_samples = 0;

  for (uint16_t i = 0; i < kSamplesPerRead; ++i) {
    // Wait briefly for a color reading to be available
    unsigned long start = millis();
    while (!APDS.colorAvailable()) {
      if (millis() - start > 50) break;  // small timeout
      delay(1);
    }

    int r = 0, g = 0, b = 0;
    APDS.readColor(r, g, b);  // Library reads current RGB; no explicit return status

    // Treat negative readings (shouldn't occur) as invalid
    if (r >= 0 && g >= 0 && b >= 0) {
      sum_r += (uint32_t)r;
      sum_g += (uint32_t)g;
      sum_b += (uint32_t)b;
      ++valid_samples;
    }

    delay(kInterSampleDelayMs);
  }

  if (valid_samples == 0) {
    return false;
  }

  avg_r = (float)sum_r / (float)valid_samples;
  avg_g = (float)sum_g / (float)valid_samples;
  avg_b = (float)sum_b / (float)valid_samples;
  return true;
}

// Preprocess: average, normalize by sum (RGB ratio), handle zero-sum
// Outputs features[3] in [0,1]
static void preprocessRGB(float features[kNumFeatures]) {
  float r = 0.0f, g = 0.0f, b = 0.0f;
  bool ok = readAveragedRawRGB(r, g, b);

  if (!ok) {
    // Could not read; reuse last valid or zeros
    if (g_has_last_valid) {
      for (int i = 0; i < kNumFeatures; ++i) {
        features[i] = g_last_valid_rgb[i];
      }
    } else {
      features[0] = features[1] = features[2] = 0.0f;
    }
    return;
  }

  float sum = r + g + b;
  if (sum <= 0.0f) {
    // Zero-sum, reuse last valid or zeros
    if (g_has_last_valid) {
      for (int i = 0; i < kNumFeatures; ++i) {
        features[i] = g_last_valid_rgb[i];
      }
    } else {
      features[0] = features[1] = features[2] = 0.0f;
    }
    return;
  }

  features[0] = clip01(r / sum);
  features[1] = clip01(g / sum);
  features[2] = clip01(b / sum);

  // Store as last valid
  for (int i = 0; i < kNumFeatures; ++i) {
    g_last_valid_rgb[i] = features[i];
  }
  g_has_last_valid = true;
}

// Copy preprocessed features into input tensor (supports float32 or uint8)
static void copyToInputTensor(const float features[kNumFeatures]) {
  if (!g_input) return;

  if (g_input->type == kTfLiteFloat32) {
    float* dst = g_input->data.f;
    for (int i = 0; i < kNumFeatures; ++i) {
      dst[i] = features[i];
    }
  } else if (g_input->type == kTfLiteUInt8) {
    uint8_t* dst = g_input->data.uint8;
    float scale = g_input->params.scale;
    int zp = g_input->params.zero_point;
    for (int i = 0; i < kNumFeatures; ++i) {
      dst[i] = quantize_uint8(features[i], scale, zp);
    }
  } else {
    // Unsupported type; zero-fill to be safe
    memset(g_input->data.raw, 0, g_input->bytes);
  }
}

// Argmax over output tensor (supports uint8 or float32)
static int argmaxOutput(int& confidence_out) {
  confidence_out = 0;
  if (!g_output) return 0;

  int best_index = 0;

  if (g_output->type == kTfLiteUInt8) {
    const uint8_t* data = g_output->data.uint8;
    uint8_t best_val = data[0];
    best_index = 0;
    for (int i = 1; i < kNumClasses; ++i) {
      if (data[i] > best_val) {
        best_val = data[i];
        best_index = i;
      }
    }
    confidence_out = (int)best_val;  // 0..255
  } else if (g_output->type == kTfLiteFloat32) {
    const float* data = g_output->data.f;
    float best_val = data[0];
    best_index = 0;
    for (int i = 1; i < kNumClasses; ++i) {
      if (data[i] > best_val) {
        best_val = data[i];
        best_index = i;
      }
    }
    // Map to 0..255-like scale for consistency
    float v = data[best_index];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    confidence_out = (int)lroundf(v * 255.0f);
  } else {
    // Unsupported type; default to class 0
    best_index = 0;
    confidence_out = 0;
  }

  return best_index;
}

// ---------- Arduino Lifecycle ----------
void setup() {
  Serial.begin(kBaudRate);
  while (!Serial && millis() < 4000) {
    ; // wait briefly for Serial
  }

  // Initialize sensor
  if (!APDS.begin()) {
    Serial.println("APDS-9960 init failed!");
    while (true) {
      delay(100);
    }
  }

  // Load model
  g_tflite_model = tflite::GetModel(model);
  if (g_tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema ");
    Serial.print(g_tflite_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) {
      delay(100);
    }
  }

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      g_tflite_model, g_resolver, g_tensor_arena, kTensorArenaSize, error_reporter);
  g_interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = g_interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) {
      delay(100);
    }
  }

  // Cache tensor pointers
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  // Basic input/output checks
  if (g_input->dims->size < 2 || g_input->dims->data[g_input->dims->size - 1] != kNumFeatures) {
    Serial.println("Unexpected input tensor shape");
  }
  if (g_output->dims->size < 2 || g_output->dims->data[g_output->dims->size - 1] != kNumClasses) {
    Serial.println("Unexpected output tensor shape");
  }

  // Warm-up inferences (optional)
  for (uint8_t i = 0; i < kWarmupInferences; ++i) {
    float features[kNumFeatures] = {0.0f, 0.0f, 0.0f};
    preprocessRGB(features);
    copyToInputTensor(features);
    if (g_interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Warmup Invoke failed");
      break;
    }
    delay(10);
  }

  Serial.println("Color Object Classifier ready.");
}

void loop() {
  // 1) Preprocessing
  float features[kNumFeatures] = {0.0f, 0.0f, 0.0f};
  preprocessRGB(features);

  // 2) Copy to model input
  copyToInputTensor(features);

  // 3) Inference
  TfLiteStatus invoke_status = g_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    delay(kInferenceIntervalMs);
    return;
  }

  // 4) Postprocessing: argmax + print label + emoji
  int confidence = 0;
  int idx = argmaxOutput(confidence);

  // Print format: label_emoji_only
  Serial.print(kLabels[idx]);
  Serial.print(" ");
  Serial.println(kEmojis[idx]);

  delay(kInferenceIntervalMs);
}