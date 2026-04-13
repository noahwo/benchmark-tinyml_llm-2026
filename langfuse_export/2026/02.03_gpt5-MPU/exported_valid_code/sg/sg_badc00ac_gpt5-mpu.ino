#include <Arduino.h>

// Programming Guidelines Phase 1.1: Include Necessary Libraries (TensorFlow Lite base first)
#include <TensorFlowLite.h>
#include "model.h"
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <Arduino_APDS9960.h>
#include <ArduinoBLE.h>      // Included as per software libraries list (not used in this sketch)

// Application/Device configuration
static const uint32_t kBaudRate = 9600;
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// Programming Guidelines Phase 1.2: Declare Variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;  // renamed to avoid conflict with byte array symbol from model.h
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Programming Guidelines Phase 1.3: Define Tensor Arena
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Helper: clamp float to [0,1]
static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

// Phase 2.1 + 2.2: Sensor Setup + Preprocessing helper
bool readNormalizedRGB(float rgb[3]) {
  if (!APDS.colorAvailable()) {
    return false;
  }
  int r = 0, g = 0, b = 0, c = 0;
  // Arduino_APDS9960: readColor returns RGBC values
  APDS.readColor(r, g, b, c);

  // Normalize to sum=1 to match dataset scale similar to given stats
  float rf = static_cast<float>(r);
  float gf = static_cast<float>(g);
  float bf = static_cast<float>(b);
  float sum = rf + gf + bf;
  if (sum <= 0.0f) {
    return false;
  }
  rgb[0] = clamp01(rf / sum); // Red
  rgb[1] = clamp01(gf / sum); // Green
  rgb[2] = clamp01(bf / sum); // Blue
  return true;
}

// Handle flexible I/O dtypes according to model metadata
void copyInputData(const float in[3]) {
  if (input->type == kTfLiteFloat32) {
    input->data.f[0] = in[0];
    input->data.f[1] = in[1];
    input->data.f[2] = in[2];
  } else if (input->type == kTfLiteUInt8) {
    const float scale = input->params.scale;
    const int zero_point = input->params.zero_point;
    input->data.uint8[0] = static_cast<uint8_t>(roundf(in[0] / scale) + zero_point);
    input->data.uint8[1] = static_cast<uint8_t>(roundf(in[1] / scale) + zero_point);
    input->data.uint8[2] = static_cast<uint8_t>(roundf(in[2] / scale) + zero_point);
  } else if (input->type == kTfLiteInt8) {
    const float scale = input->params.scale;
    const int zero_point = input->params.zero_point;
    input->data.int8[0] = static_cast<int8_t>(roundf(in[0] / scale) + zero_point);
    input->data.int8[1] = static_cast<int8_t>(roundf(in[1] / scale) + zero_point);
    input->data.int8[2] = static_cast<int8_t>(roundf(in[2] / scale) + zero_point);
  }
}

void getOutputScores(float scores[3]) {
  if (output->type == kTfLiteFloat32) {
    scores[0] = output->data.f[0];
    scores[1] = output->data.f[1];
    scores[2] = output->data.f[2];
  } else if (output->type == kTfLiteUInt8) {
    const float scale = output->params.scale;
    const int zero_point = output->params.zero_point;
    scores[0] = (static_cast<int>(output->data.uint8[0]) - zero_point) * scale;
    scores[1] = (static_cast<int>(output->data.uint8[1]) - zero_point) * scale;
    scores[2] = (static_cast<int>(output->data.uint8[2]) - zero_point) * scale;
  } else if (output->type == kTfLiteInt8) {
    const float scale = output->params.scale;
    const int zero_point = output->params.zero_point;
    scores[0] = (static_cast<int>(output->data.int8[0]) - zero_point) * scale;
    scores[1] = (static_cast<int>(output->data.int8[1]) - zero_point) * scale;
    scores[2] = (static_cast<int>(output->data.int8[2]) - zero_point) * scale;
  } else {
    scores[0] = scores[1] = scores[2] = 0.0f;
  }
}

int argmax3(const float s[3]) {
  int idx = 0;
  float best = s[0];
  if (s[1] > best) { best = s[1]; idx = 1; }
  if (s[2] > best) { idx = 2; }
  return idx;
}

void setup() {
  // Phase 1.9: Set Up Other Relevant Parts
  Serial.begin(kBaudRate);
  while (!Serial) { delay(10); }

  // Initialize color sensor
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 color sensor.");
    while (1) { delay(1000); }
  }

  // Phase 1.2: Declare and configure Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the Model
  // 'model' here refers to the byte array defined in model.h
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema "); Serial.print(tfl_model->version());
    Serial.print(" not equal to supported "); Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  // Phase 1.5: Resolve Operators
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate Memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  // Phase 1.8: Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Object Classifier by Color - Ready");
}

void loop() {
  // Phase 2: Preprocessing: Read and normalize RGB data
  float features[3];
  if (!readNormalizedRGB(features)) {
    delay(5);
    return;
  }

  // Phase 3.1: Copy Data to input tensor
  copyInputData(features);

  // Phase 3.2: Invoke Interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    delay(50);
    return;
  }

  // Phase 4.1: Process Output
  float scores[3];
  getOutputScores(scores);
  int predicted = argmax3(scores);

  // Phase 4.2: Execute Application Behavior - print to Serial with emojis
  Serial.print("RGB(norm): R=");
  Serial.print(features[0], 3);
  Serial.print(" G=");
  Serial.print(features[1], 3);
  Serial.print(" B=");
  Serial.print(features[2], 3);
  Serial.print(" | Predicted: ");
  Serial.print(kClassNames[predicted]);
  Serial.print(" ");
  Serial.print(kClassEmojis[predicted]);
  Serial.print(" | Scores: [");
  Serial.print(scores[0], 3);
  Serial.print(", ");
  Serial.print(scores[1], 3);
  Serial.print(", ");
  Serial.print(scores[2], 3);
  Serial.println("]");

  delay(150);
}