#include <Arduino.h>

// Phase 1.1: Include Necessary Libraries (TensorFlow Lite base first)
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino_APDS9960.h>
#include <Wire.h>

// Include the model file (required)
#include "model.h"

// Phase 1.2: Declare Variables
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model_ptr = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Phase 1.3: Define Tensor Arena
  constexpr int kTensorArenaSize = 8192;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];

  // Classification labels and emojis
  const char* kClasses[3] = {"Apple", "Banana", "Orange"};
  const char* kEmojis[3]  = {"🍎",    "🍌",     "🍊"};
}

// Utility: Map output tensor to float values handling both float and uint8 quantized
static float read_output_value(const TfLiteTensor* t, int idx) {
  if (t->type == kTfLiteFloat32) {
    return t->data.f[idx];
  } else if (t->type == kTfLiteUInt8) {
    // Dequantize to float
    const float scale = t->params.scale;
    const int32_t zero_point = t->params.zero_point;
    const uint8_t v = t->data.uint8[idx];
    return scale * (static_cast<int32_t>(v) - zero_point);
  } else if (t->type == kTfLiteInt8) {
    const float scale = t->params.scale;
    const int32_t zero_point = t->params.zero_point;
    const int8_t v = t->data.int8[idx];
    return scale * (static_cast<int32_t>(v) - zero_point);
  }
  return 0.0f;
}

// Find argmax and max value
static int argmax3(const float* vals, float* out_max_val) {
  int max_idx = 0;
  float max_val = vals[0];
  for (int i = 1; i < 3; ++i) {
    if (vals[i] > max_val) {
      max_val = vals[i];
      max_idx = i;
    }
  }
  if (out_max_val) *out_max_val = max_val;
  return max_idx;
}

void setup() {
  // Phase 1.9: Other setup (I/O)
  Serial.begin(9600);
  while (!Serial) { delay(10); }
  Serial.println("Object Classifier by Color (APDS9960 + TFLite Micro)");
  Serial.println("Initializing...");

  // Initialize color sensor (Phase 2.1)
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor.");
    while (1) { delay(1000); }
  } else {
    Serial.println("APDS9960 initialized.");
  }

  // Phase 1.2: Setup Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the Model
  // model and model_len are expected to be defined in model.h
  extern const unsigned char model[];    // Provided by model.h
  extern const int model_len;            // Provided by model.h
  model_ptr = tflite::GetModel(model);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported %d.",
                           model_ptr->version(), TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  // Phase 1.5: Resolve Operators
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model_ptr, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate Memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  // Phase 1.8: Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor: expect float32 and 1x3
  bool input_ok = true;
  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Unexpected input type. Expected float32.");
    input_ok = false;
  }
  if (input->dims->size < 2 || input->dims->data[input->dims->size - 1] != 3) {
    error_reporter->Report("Unexpected input dims; expected last dimension 3.");
    input_ok = false;
  }
  if (!input_ok) {
    Serial.println("Input tensor mismatch with application specification.");
    while (1) { delay(1000); }
  }

  // Basic info
  Serial.print("Tensor arena bytes used: ");
  Serial.println(kTensorArenaSize);
  Serial.println("Initialization complete. Move a colored object in front of the sensor.");
}

void loop() {
  // Phase 2.1: Acquire sensor sample
  int r = 0, g = 0, b = 0;
  // Wait for a color sample to be ready
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }
  // Read color; some library versions provide readColor(r,g,b) or readColor(r,g,b,c)
  if (!APDS.readColor(r, g, b)) {
    // If read fails, skip this iteration
    return;
  }

  // Phase 2.2: Preprocess (normalize to match dataset: Red+Green+Blue ~= 1)
  float rf = static_cast<float>(r);
  float gf = static_cast<float>(g);
  float bf = static_cast<float>(b);
  float sum = rf + gf + bf;
  if (sum <= 0.0f) {
    // Avoid divide-by-zero; try again
    return;
  }
  float features[3];
  features[0] = rf / sum;  // Red
  features[1] = gf / sum;  // Green
  features[2] = bf / sum;  // Blue

  // Optional: clamp to [0,1]
  for (int i = 0; i < 3; ++i) {
    if (features[i] < 0.0f) features[i] = 0.0f;
    if (features[i] > 1.0f) features[i] = 1.0f;
  }

  // Phase 3.1: Copy data into input tensor
  input = interpreter->input(0);
  for (int i = 0; i < 3; ++i) {
    input->data.f[i] = features[i];
  }

  // Phase 3.2: Invoke Interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    // Inference failed; skip this loop
    return;
  }

  // Phase 4.1: Process Output
  output = interpreter->output(0);
  float scores[3];
  for (int i = 0; i < 3; ++i) {
    scores[i] = read_output_value(output, i);
  }
  float max_score = 0.f;
  int idx = argmax3(scores, &max_score);

  // Phase 4.2: Execute Application Behavior (Serial output with emoji)
  Serial.print("RGB norm: [");
  Serial.print(features[0], 3); Serial.print(", ");
  Serial.print(features[1], 3); Serial.print(", ");
  Serial.print(features[2], 3); Serial.print("]  ->  ");

  Serial.print("Class: ");
  Serial.print(kClasses[idx]);
  Serial.print(" ");
  Serial.print(kEmojis[idx]);
  Serial.print("  Scores: [");
  Serial.print(scores[0], 3); Serial.print(", ");
  Serial.print(scores[1], 3); Serial.print(", ");
  Serial.print(scores[2], 3); Serial.println("]");

  delay(150);
}