#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <ArduinoBLE.h>
#include <Arduino_LPS22HB.h>

#include "model.h"

// Programming Guidelines Phase 1: Initialization
// 1.1 Include Necessary Libraries
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// 1.2 Declare Variables
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tfl_model = nullptr;  // renamed to avoid conflict with model array in model.h
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // 1.3 Define Tensor Arena
  constexpr int kTensorArenaSize = 16384;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  // Classification labels and emojis
  const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
  const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};
}

// Helper: Read and normalize RGB from APDS9960 into range [0,1] using sum normalization
bool readNormalizedRGB(float rgb[3]) {
  int r = 0, g = 0, b = 0;
  // Wait for a fresh color sample
  unsigned long start = millis();
  while (!APDS.colorAvailable()) {
    if (millis() - start > 100) break;
    delay(5);
  }
  if (!APDS.colorAvailable()) {
    return false;
  }
  APDS.readColor(r, g, b);
  const float rf = static_cast<float>(r);
  const float gf = static_cast<float>(g);
  const float bf = static_cast<float>(b);
  const float sum = rf + gf + bf;
  if (sum <= 0.0f) {
    return false;
  }
  // Sum normalization to match dataset scale (approximately 0.5-0.6 for Red channel, etc.)
  rgb[0] = rf / sum;
  rgb[1] = gf / sum;
  rgb[2] = bf / sum;
  return true;
}

// Helper: Argmax for 3-class output
int argmax3_uint8(const uint8_t* v) {
  int idx = 0;
  uint8_t best = v[0];
  if (v[1] > best) { best = v[1]; idx = 1; }
  if (v[2] > best) { best = v[2]; idx = 2; }
  return idx;
}

int argmax3_float(const float* v) {
  int idx = 0;
  float best = v[0];
  if (v[1] > best) { best = v[1]; idx = 1; }
  if (v[2] > best) { best = v[2]; idx = 2; }
  return idx;
}

void setup() {
  // Serial Initialization
  Serial.begin(9600);
  while (!Serial) { ; }

  Serial.println("Object Classifier by Color (TinyML)");
  Serial.println("Initializing...");

  // Sensor bus
  Wire.begin();

  // 1.9 Initialize other relevant parts (sensor)
  if (!APDS.begin()) {
    Serial.println("Error: Failed to initialize APDS9960 sensor.");
    while (1) { delay(100); }
  }
  // Increase color gain to improve measurement stability (optional)
  // The Arduino_APDS9960 library does not expose direct gain control for color;
  // ensure ambient light is sufficient for stable readings.
  Serial.println("APDS9960 initialized.");

  // 1.4 Load the model
  // model.h provides the flatbuffer array symbol 'model'.
  tfl_model = tflite::GetModel(::model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch. Expected: ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(" Got: ");
    Serial.println(tfl_model->version());
    while (1) { delay(100); }
  }

  // 1.5 Resolve Operators
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  static tflite::AllOpsResolver resolver;

  // 1.6 Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // 1.7 Allocate Memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("Error: AllocateTensors() failed.");
    while (1) { delay(100); }
  }

  // 1.8 Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Check input shape and type: expected [1, 3], float32
  bool input_ok = (input != nullptr) &&
                  (input->type == kTfLiteFloat32) &&
                  (input->dims != nullptr) &&
                  (input->dims->size >= 2) &&
                  (input->dims->data[input->dims->size - 1] == 3);
  if (!input_ok) {
    Serial.println("Error: Input tensor mismatch. Expecting float32 with 3 features.");
    while (1) { delay(100); }
  }

  // Check output shape: expected 3 classes, dtype can be uint8 (quantized) or float32
  bool output_ok = (output != nullptr) &&
                   (output->dims != nullptr) &&
                   (output->dims->size >= 2) &&
                   (output->dims->data[output->dims->size - 1] == 3) &&
                   (output->type == kTfLiteUInt8 || output->type == kTfLiteFloat32);
  if (!output_ok) {
    Serial.println("Error: Output tensor mismatch. Expecting 3 classes (uint8 or float32).");
    while (1) { delay(100); }
  }

  Serial.println("Initialization complete.");
  Serial.println("Reading RGB and running inference...");
}

void loop() {
  // Phase 2: Preprocessing - Sensor Setup and feature extraction
  float rgb[3];
  if (!readNormalizedRGB(rgb)) {
    // If no new data, wait a bit
    delay(20);
    return;
  }

  // Optional clamping for safety
  for (int i = 0; i < 3; i++) {
    if (rgb[i] < 0.0f) rgb[i] = 0.0f;
    if (rgb[i] > 1.0f) rgb[i] = 1.0f;
  }

  // Phase 3: Inference
  // 3.1 Data Copy
  input->data.f[0] = rgb[0]; // Red
  input->data.f[1] = rgb[1]; // Green
  input->data.f[2] = rgb[2]; // Blue

  // 3.2 Invoke Interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error: Inference failed.");
    delay(100);
    return;
  }

  // Phase 4: Postprocessing
  int class_idx = 0;
  if (output->type == kTfLiteUInt8) {
    const uint8_t* logits = output->data.uint8;
    class_idx = argmax3_uint8(logits);
    Serial.print("Scores (uint8): [");
    Serial.print(logits[0]); Serial.print(", ");
    Serial.print(logits[1]); Serial.print(", ");
    Serial.print(logits[2]); Serial.println("]");
  } else { // kTfLiteFloat32
    const float* logits = output->data.f;
    class_idx = argmax3_float(logits);
    Serial.print("Scores (float): [");
    Serial.print(logits[0], 4); Serial.print(", ");
    Serial.print(logits[1], 4); Serial.print(", ");
    Serial.print(logits[2], 4); Serial.println("]");
  }

  // 4.2 Execute Application Behavior: print class with emoji and normalized RGB input
  Serial.print("Input RGB(norm): R=");
  Serial.print(rgb[0], 3);
  Serial.print(" G=");
  Serial.print(rgb[1], 3);
  Serial.print(" B=");
  Serial.print(rgb[2], 3);
  Serial.println();

  Serial.print("Prediction: ");
  Serial.print(kClassNames[class_idx]);
  Serial.print(" ");
  Serial.println(kClassEmojis[class_idx]);

  Serial.println("-----------------------------");
  delay(250);
}