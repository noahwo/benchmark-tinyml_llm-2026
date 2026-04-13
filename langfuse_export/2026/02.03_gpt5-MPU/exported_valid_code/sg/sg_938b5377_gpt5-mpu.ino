#include <Arduino.h>
#include <TensorFlowLite.h>  // Must be included before any TFLM micro headers
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <ArduinoBLE.h>
#include <Arduino_LPS22HB.h>

#include "model.h"  // Contains the compiled TFLite model flatbuffer

// Phase 1: Initialization - Declare TFLM core variables
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model_ptr = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Define the tensor arena (ensure it's sufficiently large but not wasteful)
  constexpr int kTensorArenaSize = 8192;
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// Attempt to resolve model pointer from common symbol names without guessing a single one
extern const unsigned char g_model[] __attribute__((weak));
extern const unsigned char model_tflite[] __attribute__((weak));
static const unsigned char* ResolveModelPointer() {
  if (&g_model != nullptr && g_model != nullptr) return g_model;
  if (&model != nullptr && model != nullptr) return model;
  if (&model_tflite != nullptr && model_tflite != nullptr) return model_tflite;
  return nullptr;
}

// Application constants
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// Helper: print tensor info (for debugging)
static void PrintInputInfo() {
  if (!input) return;
  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.print("Input dims: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print("x");
  }
  Serial.println();
}
static void PrintOutputInfo() {
  if (!output) return;
  Serial.print("Output type: ");
  Serial.println(output->type);
  Serial.print("Output dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print("x");
  }
  Serial.println();
}

// Phase 2: Sensor setup
static bool InitColorSensor() {
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 color sensor.");
    return false;
  }
  // Optional: configure if needed (defaults generally work)
  // APDS.setGain(); APDS.setColorIntegrationTime(); etc.
  Serial.println("APDS9960 initialized.");
  return true;
}

// Preprocess raw RGB to normalized floats matching typical dataset scaling (r/(r+g+b), etc.)
static bool PreprocessRGB(int r, int g, int b, float out_features[3]) {
  long sum = (long)r + (long)g + (long)b;
  if (sum <= 0) {
    return false;
  }
  out_features[0] = (float)r / (float)sum;  // Red
  out_features[1] = (float)g / (float)sum;  // Green
  out_features[2] = (float)b / (float)sum;  // Blue
  // Clamp to [0,1]
  for (int i = 0; i < 3; i++) {
    if (out_features[i] < 0.0f) out_features[i] = 0.0f;
    if (out_features[i] > 1.0f) out_features[i] = 1.0f;
  }
  return true;
}

// Phase 4: Postprocessing - Argmax over uint8 outputs
static int ArgMaxUint8(const uint8_t* scores, int len) {
  int best_idx = 0;
  uint8_t best_val = scores[0];
  for (int i = 1; i < len; i++) {
    if (scores[i] > best_val) {
      best_val = scores[i];
      best_idx = i;
    }
  }
  return best_idx;
}

void setup() {
  // Serial interface
  Serial.begin(9600);
  while (!Serial) { delay(10); }
  Serial.println("Object Classifier by Color - TinyML Inference");

  // Initialize I2C and optional peripherals
  Wire.begin();
  // BLE and Pressure sensor libraries are included; initialization is optional for this app
  // BLE.begin(); BARO.begin();

  // Phase 2.1: Sensor setup
  if (!InitColorSensor()) {
    Serial.println("Halting due to sensor init failure.");
    while (true) delay(1000);
  }

  // Phase 1: Initialization of TFLM
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  const unsigned char* model_data = ResolveModelPointer();
  if (model_data == nullptr) {
    Serial.println("ERROR: Model data pointer not found in model.h (expected one of g_model, model, model_tflite).");
    while (true) delay(1000);
  }

  model_ptr = tflite::GetModel(model_data);
  if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema version mismatch. Found: ");
    Serial.print(model_ptr->version());
    Serial.print(" Expected: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) delay(1000);
  }

  // Phase 1.5: Resolve operators (use AllOpsResolver as architecture unknown)
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(model_ptr, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (true) delay(1000);
  }

  // Phase 1.8: Define model inputs/outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor: expected 1x3 float32
  bool input_ok = (input->type == kTfLiteFloat32);
  if (input->dims->size == 2) {
    input_ok = input_ok && (input->dims->data[0] == 1) && (input->dims->data[1] == 3);
  } else if (input->dims->size == 1) {
    // Some models may flatten to [3]
    input_ok = input_ok && (input->dims->data[0] == 3);
  } else {
    input_ok = false;
  }

  // Validate output tensor: expected 1x3 uint8
  bool output_ok = (output->type == kTfLiteUInt8);
  if (output->dims->size == 2) {
    output_ok = output_ok && (output->dims->data[0] == 1) && (output->dims->data[1] == 3);
  } else if (output->dims->size == 1) {
    output_ok = output_ok && (output->dims->data[0] == 3);
  } else {
    output_ok = false;
  }

  if (!input_ok || !output_ok) {
    Serial.println("ERROR: Tensor shapes/types mismatch. Expected input: float32 [1x3], output: uint8 [1x3].");
    PrintInputInfo();
    PrintOutputInfo();
    while (true) delay(1000);
  }

  Serial.println("TFLM initialized successfully.");
  PrintInputInfo();
  PrintOutputInfo();
}

void loop() {
  // Phase 2.1: Acquire sensor data
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int r = 0, g = 0, b = 0;
  if (!APDS.readColor(r, g, b)) {
    // Read failed; try again
    delay(5);
    return;
  }

  // Phase 2.2: Preprocess (normalize RGB)
  float features[3];
  if (!PreprocessRGB(r, g, b, features)) {
    // Avoid feeding invalid data
    delay(5);
    return;
  }

  // Phase 3.1: Copy data to input tensor
  input->data.f[0] = features[0];  // Red
  input->data.f[1] = features[1];  // Green
  input->data.f[2] = features[2];  // Blue

  // Phase 3.2: Invoke interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(100);
    return;
  }

  // Phase 4.1: Process Output (uint8 scores)
  const uint8_t* scores = output->data.uint8;
  int predicted = ArgMaxUint8(scores, 3);

  // Phase 4.2: Execute Application Behavior (print classification with emoji)
  Serial.print("Color RGB (raw): ");
  Serial.print(r); Serial.print(", ");
  Serial.print(g); Serial.print(", ");
  Serial.print(b);
  Serial.print(" | Features (R,G,B): ");
  Serial.print(features[0], 3); Serial.print(", ");
  Serial.print(features[1], 3); Serial.print(", ");
  Serial.print(features[2], 3);

  Serial.print(" | Scores: [");
  Serial.print((int)scores[0]); Serial.print(", ");
  Serial.print((int)scores[1]); Serial.print(", ");
  Serial.print((int)scores[2]); Serial.print("]");

  Serial.print(" | Class: ");
  Serial.print(kClassNames[predicted]);
  Serial.print(" ");
  Serial.println(kClassEmojis[predicted]);

  delay(150);
}