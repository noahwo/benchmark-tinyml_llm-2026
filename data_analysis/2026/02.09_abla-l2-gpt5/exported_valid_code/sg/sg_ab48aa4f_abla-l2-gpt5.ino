/*
  Color Object Classifier
  Board: Arduino Nano 33 BLE Sense
  Sensor: APDS-9960 (RGB)
  Task: Classify objects (Apple, Banana, Orange) from RGB color with TensorFlow Lite Micro
  Output: Unicode emoji over Serial
*/

#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "model.h"  // contains: const unsigned char model[] = {...}

// -------------------- Application Constants --------------------
static const int kSampleRateHz = 5;           // 5 Hz as specified
static const uint32_t kSamplePeriodMs = 1000 / kSampleRateHz;
static const int kNumFeatures = 3;            // Red, Green, Blue
static const int kNumClasses = 3;             // Apple, Banana, Orange
static const size_t kTensorArenaSize = 16384; // Specified tensor arena size

// Class labels and emojis
static const char* kClassNames[kNumClasses] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};

// -------------------- TFLM Globals --------------------
namespace {
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

tflite::AllOpsResolver resolver;

const tflite::Model* tflm_model = nullptr;          // Avoid name clash with byte array 'model' from model.h
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// -------------------- Utility: Clamp --------------------
static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// -------------------- Sensor Read + Preprocessing --------------------
/*
  Reads RGB from APDS-9960 and normalizes to [0,1] as per dataset spec.
  Returns true on success and fills features[0..2] in order [Red, Green, Blue].
*/
bool readNormalizedRGB(float features[kNumFeatures]) {
  if (!APDS.colorAvailable()) {
    return false;
  }
  int r = 0, g = 0, b = 0;
  APDS.readColor(r, g, b);

  // Library typically returns 0..255; normalize to [0,1]
  features[0] = clampf((float)r / 255.0f, 0.0f, 1.0f); // Red
  features[1] = clampf((float)g / 255.0f, 0.0f, 1.0f); // Green
  features[2] = clampf((float)b / 255.0f, 0.0f, 1.0f); // Blue
  return true;
}

// -------------------- Optional: Quantize helper for uint8 inputs --------------------
static inline uint8_t quantizeUInt8(float x, float scale, int zero_point) {
  int32_t q = (int32_t)roundf(x / scale) + zero_point;
  if (q < 0) q = 0;
  if (q > 255) q = 255;
  return (uint8_t)q;
}

// -------------------- Argmax --------------------
int argmax_float(const float* data, int len) {
  int idx = 0;
  float best = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

int argmax_uint8(const uint8_t* data, int len) {
  int idx = 0;
  uint8_t best = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

// -------------------- Setup --------------------
void setup() {
  Serial.begin(9600);
  while (!Serial) { /* wait for USB serial */ }

  Serial.println("Color Object Classifier (Apple 🍎, Banana 🍌, Orange 🍊)");
  Serial.println("Initializing APDS-9960...");

  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS-9960 sensor.");
    while (1) { delay(1000); }
  }
  Serial.println("APDS-9960 OK");

  // Initialize TFLM
  tflm_model = tflite::GetModel(model); // 'model' is the byte array from model.h
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema ");
    Serial.print(tflm_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Basic sanity checks according to spec
  if (input->dims->size != 2 || input->dims->data[0] != 1 || input->dims->data[1] != kNumFeatures) {
    Serial.println("WARNING: Unexpected input tensor shape. Expected [1,3].");
  }
  if (output->dims->size < 2 || output->dims->data[0] != 1 || output->dims->data[1] != kNumClasses) {
    Serial.println("WARNING: Unexpected output tensor shape. Expected [1,3].");
  }

  Serial.println("Setup complete. Sampling at 5 Hz.\n");
}

// -------------------- Loop --------------------
void loop() {
  static uint32_t last_ms = 0;
  const uint32_t now = millis();
  if (now - last_ms < kSamplePeriodMs) {
    delay(5);
    return;
  }
  last_ms = now;

  float features[kNumFeatures];
  if (!readNormalizedRGB(features)) {
    // No new color sample yet; keep waiting within sample rate window
    return;
  }

  // Preprocessing: values are already normalized and clipped to [0,1]
  // Copy data to input tensor (supports float32 or uint8 inputs)
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < kNumFeatures; ++i) {
      input->data.f[i] = features[i];
    }
  } else if (input->type == kTfLiteUInt8) {
    const float scale = input->params.scale;
    const int zp = input->params.zero_point;
    for (int i = 0; i < kNumFeatures; ++i) {
      input->data.uint8[i] = quantizeUInt8(features[i], scale, zp);
    }
  } else {
    Serial.println("ERROR: Unsupported input tensor type.");
    delay(200);
    return;
  }

  // Inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(200);
    return;
  }

  // Postprocessing: argmax and optional confidence
  int pred_idx = 0;
  float conf = 0.0f;

  if (output->type == kTfLiteFloat32) {
    pred_idx = argmax_float(output->data.f, kNumClasses);
    // Confidence best effort: if softmax, values in [0,1]; otherwise just take max
    conf = output->data.f[pred_idx];
    conf = clampf(conf, 0.0f, 1.0f);
  } else if (output->type == kTfLiteUInt8) {
    pred_idx = argmax_uint8(output->data.uint8, kNumClasses);
    // Dequantize selected score to [0,1] if scale/zero_point indicate probabilities
    const float scale = output->params.scale;
    const int zp = output->params.zero_point;
    conf = scale * (static_cast<int>(output->data.uint8[pred_idx]) - zp);
    // Clamp to [0,1] for display convenience
    conf = clampf(conf, 0.0f, 1.0f);
  } else {
    Serial.println("ERROR: Unsupported output tensor type.");
    delay(200);
    return;
  }

  // Emit result over Serial with emoji
  const char* label = (pred_idx >= 0 && pred_idx < kNumClasses) ? kClassNames[pred_idx] : "Unknown";
  const char* emoji = (pred_idx >= 0 && pred_idx < kNumClasses) ? kClassEmojis[pred_idx] : "❓";

  Serial.print("RGB: [");
  Serial.print(features[0], 3); Serial.print(", ");
  Serial.print(features[1], 3); Serial.print(", ");
  Serial.print(features[2], 3); Serial.print("]  ->  ");
  Serial.print(label);
  Serial.print(" ");
  Serial.print(emoji);
  Serial.print("  (conf ~ ");
  Serial.print(conf, 2);
  Serial.println(")");

  // Maintain ~5 Hz
  // (We already enforced period at loop start; an extra small delay smooths Serial)
  delay(5);
}