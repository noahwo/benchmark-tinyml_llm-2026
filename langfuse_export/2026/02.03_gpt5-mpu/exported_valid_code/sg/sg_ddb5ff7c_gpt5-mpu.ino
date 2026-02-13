#include <Arduino.h>

// Phase 1.1: Include Necessary Libraries (TensorFlowLite base first)
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Sensors and I/O
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <Arduino_LPS22HB.h>
#include <Arduino_HTS221.h>

// Include the model file (path requirement)
#include "model.h"

// Application constants
static const int kSerialBaud = 9600;
static const int kNumInputFeatures = 3;   // Red, Green, Blue
static const int kNumClasses = 3;         // Apple, Banana, Orange

// Tensor arena size from specifications
constexpr int kTensorArenaSize = 16384;
static uint8_t tensor_arena[kTensorArenaSize];

// Phase 1.2: Declare critical TFLM variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflm_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Class labels and emojis
const char* kClassNames[kNumClasses] = { "Apple", "Banana", "Orange" };
const char* kClassEmojis[kNumClasses] = { "🍎", "🍌", "🍊" };

// Utility: ArgMax for classification
static int ArgMax(const float* data, int len) {
  int idx = 0;
  float max_v = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > max_v) {
      max_v = data[i];
      idx = i;
    }
  }
  return idx;
}

// Phase 2.1: Sensor Setup and helper to read normalized RGB
// Normalization strategy: chromaticity normalization r/sum, g/sum, b/sum (matches dataset ~ sums to ~1)
static bool readNormalizedRGB(float* rgb_out) {
  if (!APDS.colorAvailable()) {
    return false;
  }
  int r = 0, g = 0, b = 0;
  if (!APDS.readColor(r, g, b)) {
    return false;
  }

  // Guard against negative or zero readings
  if (r < 0) r = 0;
  if (g < 0) g = 0;
  if (b < 0) b = 0;

  long sum = static_cast<long>(r) + static_cast<long>(g) + static_cast<long>(b);
  if (sum <= 0) {
    // No light detected; provide a safe default (avoid division by zero)
    rgb_out[0] = 0.0f;
    rgb_out[1] = 0.0f;
    rgb_out[2] = 0.0f;
    return true;
  }

  rgb_out[0] = static_cast<float>(r) / static_cast<float>(sum); // Red
  rgb_out[1] = static_cast<float>(g) / static_cast<float>(sum); // Green
  rgb_out[2] = static_cast<float>(b) / static_cast<float>(sum); // Blue
  return true;
}

void setup() {
  // Serial initialization
  Serial.begin(kSerialBaud);
  while (!Serial) { /* wait for Serial */ }

  Serial.println("Object Classifier by Color - Initializing...");

  // Phase 1.9: Initialize other components (I2C/Sensors)
  Wire.begin();

  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 color sensor.");
  } else {
    Serial.println("APDS9960 initialized.");
  }

  // Phase 1.2: Create error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the model from model data
  // Note: model data symbol is provided in model.h as 'model'
  tflm_model = tflite::GetModel(model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema ");
    Serial.print(tflm_model->version());
    Serial.print(" not equal to supported version ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  // Phase 1.5: Resolve operators
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate tensor memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    return;
  }

  // Phase 1.8: Define model inputs/outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor: expecting [1,3] float32 per specification
  bool input_ok = (input != nullptr) &&
                  (input->type == kTfLiteFloat32) &&
                  (input->dims != nullptr) &&
                  (input->dims->size >= 2) &&
                  (input->dims->data[input->dims->size - 1] == kNumInputFeatures);
  if (!input_ok) {
    Serial.println("ERROR: Input tensor mismatch. Expected float32 with 3 features.");
    return;
  }

  // Validate output tensor: expecting [1,3] uint8 per specification (but support float too)
  bool output_ok = (output != nullptr) &&
                   ((output->type == kTfLiteUInt8) || (output->type == kTfLiteFloat32)) &&
                   (output->dims != nullptr) &&
                   (output->dims->size >= 2) &&
                   (output->dims->data[output->dims->size - 1] == kNumClasses);
  if (!output_ok) {
    Serial.println("ERROR: Output tensor mismatch. Expected uint8/float32 with 3 classes.");
    return;
  }

  Serial.println("Initialization complete. Starting inference loop...");
}

void loop() {
  // Phase 2: Preprocessing - Read and normalize sensor data
  float features[kNumInputFeatures];
  if (!readNormalizedRGB(features)) {
    delay(5);
    return;
  }

  // Phase 3.1: Copy data into input tensor buffer
  // Input dtype: float32 with 3 features
  for (int i = 0; i < kNumInputFeatures; ++i) {
    input->data.f[i] = features[i];
  }

  // Phase 3.2: Invoke interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(50);
    return;
  }

  // Phase 4.1: Process output
  float probs[kNumClasses];
  if (output->type == kTfLiteUInt8) {
    const float scale = output->params.scale;
    const int32_t zero_point = output->params.zero_point;
    for (int i = 0; i < kNumClasses; ++i) {
      int32_t q = static_cast<int32_t>(output->data.uint8[i]);
      probs[i] = (static_cast<float>(q) - static_cast<float>(zero_point)) * scale;
      if (probs[i] < 0.0f) probs[i] = 0.0f;
      if (probs[i] > 1.0f) probs[i] = 1.0f;
    }
  } else if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < kNumClasses; ++i) {
      probs[i] = output->data.f[i];
    }
  } else {
    // Unsupported output type
    Serial.println("ERROR: Unsupported output tensor type.");
    delay(100);
    return;
  }

  // Determine predicted class
  int pred_idx = ArgMax(probs, kNumClasses);

  // Phase 4.2: Execute application behavior - print result with emoji
  Serial.print("Input RGB (normalized): R=");
  Serial.print(features[0], 3);
  Serial.print(" G=");
  Serial.print(features[1], 3);
  Serial.print(" B=");
  Serial.print(features[2], 3);
  Serial.println();

  Serial.print("Prediction: ");
  Serial.print(kClassNames[pred_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[pred_idx]);
  Serial.print(" | Probabilities -> ");
  for (int i = 0; i < kNumClasses; ++i) {
    Serial.print(kClassNames[i]);
    Serial.print(": ");
    Serial.print(probs[i], 3);
    if (i < kNumClasses - 1) Serial.print(", ");
  }
  Serial.println();
  Serial.println("-----------------------------");

  delay(200); // Rate limit for readability
}