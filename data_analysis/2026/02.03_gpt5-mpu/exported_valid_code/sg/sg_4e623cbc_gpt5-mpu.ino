#include <Arduino.h>

// Phase 1.1: Include Necessary Libraries (Base before dependent)
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Sensors and I/O
#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <ArduinoBLE.h>

// Model header (required)
#include "model.h"

// -----------------------------------------------------------------------------
// Globals (Phase 1.2)
// -----------------------------------------------------------------------------
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;  // Renamed to avoid conflict with model.h symbol
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Phase 1.3: Tensor Arena
constexpr int kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// Application constants
const unsigned long kBaudRate = 9600;
const char* kClassNames[3]  = {"Apple", "Banana", "Orange"};
const char* kClassEmojis[3] = {"🍎",    "🍌",    "🍊"};

// Utility: safe print for float with fixed decimals
void printFloat3(float v) {
  Serial.print(v, 3);
}

// Read and normalize RGB from APDS9960 into range that sums to ~1.0
// Returns true on success, false otherwise.
bool readNormalizedRGB(float rgb[3]) {
  // Phase 2.1: Sensor Setup / Read
  int r = 0, g = 0, b = 0;

  // Try a few times to acquire a new color sample
  for (int tries = 0; tries < 10; ++tries) {
    if (APDS.colorAvailable()) {
      APDS.readColor(r, g, b);
      break;
    }
    delay(5);
  }

  // If nothing was read, still zero values; guard against divide-by-zero
  long sum = static_cast<long>(r) + static_cast<long>(g) + static_cast<long>(b);
  if (sum <= 0) {
    return false;
  }

  // Phase 2.2: Optional Feature Extraction (normalize so R+G+B ~= 1.0)
  rgb[0] = static_cast<float>(r) / static_cast<float>(sum);
  rgb[1] = static_cast<float>(g) / static_cast<float>(sum);
  rgb[2] = static_cast<float>(b) / static_cast<float>(sum);
  return true;
}

// Argmax utility for 3-element arrays (uint8)
int argmax_u8(const uint8_t* data, int len) {
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

// Argmax utility for 3-element arrays (float)
int argmax_f32(const float* data, int len) {
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

void setup() {
  // Serial init (deployment interface)
  Serial.begin(kBaudRate);
  while (!Serial && millis() < 4000) { /* wait for Serial */ }

  // Phase 1.9: Initialize peripherals
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 color sensor.");
    while (true) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // Phase 1.2: Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the model
  tfl_model = tflite::GetModel(::model);  // Use array from model.h explicitly
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported %d.",
                           tfl_model->version(), TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Phase 1.5: Resolve Operators (fallback to all ops)
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate Memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // Phase 1.8: Define Model Inputs/Outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor
  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Model input type is not float32.");
    while (true) { delay(1000); }
  }
  if (!(input->dims->size >= 2 &&
        input->dims->data[input->dims->size - 1] == 3)) {
    error_reporter->Report("Unexpected input tensor shape.");
    while (true) { delay(1000); }
  }

  Serial.println("TFLite Micro interpreter initialized.");
  Serial.println("Object Classifier by Color is ready.");
}

void loop() {
  // Acquire and preprocess data
  float rgb[3] = {0.f, 0.f, 0.f};
  if (!readNormalizedRGB(rgb)) {
    // No valid data, try again soon
    delay(20);
    return;
  }

  // Phase 3.1: Copy data to model input tensor
  input->data.f[0] = rgb[0];
  input->data.f[1] = rgb[1];
  input->data.f[2] = rgb[2];

  // Phase 3.2: Invoke Interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(50);
    return;
  }

  // Phase 4.1: Process Output
  int predicted_index = -1;

  if (output->type == kTfLiteUInt8) {
    // Quantized uint8 scores
    const uint8_t* scores = output->data.uint8;
    predicted_index = argmax_u8(scores, 3);

    // Phase 4.2: Execute Application Behavior (Serial output with emoji)
    Serial.print("RGB(norm): [");
    printFloat3(rgb[0]); Serial.print(", ");
    printFloat3(rgb[1]); Serial.print(", ");
    printFloat3(rgb[2]); Serial.print("]  Scores(uint8): [");
    Serial.print(scores[0]); Serial.print(", ");
    Serial.print(scores[1]); Serial.print(", ");
    Serial.print(scores[2]); Serial.print("]  -> ");
  } else if (output->type == kTfLiteFloat32) {
    // Float scores fallback (if model exported differently)
    const float* scores = output->data.f;
    predicted_index = argmax_f32(scores, 3);

    Serial.print("RGB(norm): [");
    printFloat3(rgb[0]); Serial.print(", ");
    printFloat3(rgb[1]); Serial.print(", ");
    printFloat3(rgb[2]); Serial.print("]  Scores(f32): [");
    printFloat3(scores[0]); Serial.print(", ");
    printFloat3(scores[1]); Serial.print(", ");
    printFloat3(scores[2]); Serial.print("]  -> ");
  } else {
    Serial.println("ERROR: Unsupported output tensor type.");
    delay(100);
    return;
  }

  if (predicted_index >= 0 && predicted_index < 3) {
    Serial.print(kClassNames[predicted_index]);
    Serial.print(" ");
    Serial.println(kClassEmojis[predicted_index]);
  } else {
    Serial.println("Unknown");
  }

  delay(150);
}