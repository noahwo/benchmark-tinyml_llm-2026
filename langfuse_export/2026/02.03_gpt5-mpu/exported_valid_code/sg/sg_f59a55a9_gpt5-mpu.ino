#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <Arduino_APDS9960.h>

#include <TensorFlowLite.h>  // Base TFLM header (must be before micro/* headers)
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model.h"  // Must include the model header with the TFLite model data

// TensorFlow Lite Micro globals (phase 1: declare variables)
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tfl_model = nullptr;  // renamed to avoid conflict with model.h symbol
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor Arena (phase 1: define tensor arena)
constexpr int kTensorArenaSize = 16384;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Application specifics
static const char* kLabelNames[3] = { "Apple", "Banana", "Orange" };
static const char* kLabelEmojis[3] = { "🍎", "🍌", "🍊" };

// Helper: find argmax and return index
int argmax_u8(const uint8_t* data, int len) {
  int max_index = 0;
  uint8_t max_val = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_index = i;
    }
  }
  return max_index;
}

int argmax_f32(const float* data, int len) {
  int max_index = 0;
  float max_val = data[0];
  for (int i = 1; i < len; ++i) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_index = i;
    }
  }
  return max_index;
}

void setup() {
  // Phase 1.1 / Deployment: Serial init
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color - Initializing...");

  // Phase 1.9: Initialize sensor
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 color sensor. Halting.");
    while (true) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // Phase 1.2: Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the model
  tfl_model = tflite::GetModel(model);  // 'model' is the byte array from model.h
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch: expected %d, got %d.",
                           TFLITE_SCHEMA_VERSION, tfl_model->version());
    while (true) { delay(1000); }
  }

  // Phase 1.5: Resolve operators (use AllOpsResolver as fallback)
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate tensor memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // Phase 1.8: Define model inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate expected tensor specs if possible
  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("Unexpected input tensor type: expected float32.");
  }
  if (output->type != kTfLiteUInt8 && output->type != kTfLiteFloat32) {
    error_reporter->Report("Unexpected output tensor type: expected uint8 or float32.");
  }

  Serial.println("Initialization complete. Starting inference loop.");
}

void loop() {
  // Phase 2.1: Sensor setup/read
  static int r = 0, g = 0, b = 0; // RGB values from APDS9960
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  // Read color values (0-255 typical range for this library)
  APDS.readColor(r, g, b);

  // Phase 2.2: Preprocessing - normalize to [0,1] to match dataset distribution
  float red   = constrain(r, 0, 255) / 255.0f;
  float green = constrain(g, 0, 255) / 255.0f;
  float blue  = constrain(b, 0, 255) / 255.0f;

  // Optional: Clamp further to [0,1] just in case
  red = red < 0.f ? 0.f : (red > 1.f ? 1.f : red);
  green = green < 0.f ? 0.f : (green > 1.f ? 1.f : green);
  blue = blue < 0.f ? 0.f : (blue > 1.f ? 1.f : blue);

  // Phase 3.1: Copy data to model input tensor
  if (input->type == kTfLiteFloat32 && input->dims->size >= 2 &&
      input->dims->data[input->dims->size - 1] >= 3) {
    float* input_data = input->data.f;
    input_data[0] = red;
    input_data[1] = green;
    input_data[2] = blue;
  } else {
    // If shape/type mismatch, halt to avoid undefined behavior
    Serial.println("Input tensor shape/type mismatch.");
    delay(500);
    return;
  }

  // Phase 3.2: Invoke the interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(100);
    return;
  }

  // Phase 4.1: Process output
  int predicted_index = -1;
  if (output->type == kTfLiteUInt8) {
    const uint8_t* out_data = output->data.uint8;
    predicted_index = argmax_u8(out_data, 3);
  } else if (output->type == kTfLiteFloat32) {
    const float* out_data = output->data.f;
    predicted_index = argmax_f32(out_data, 3);
  }

  // Safety check
  if (predicted_index < 0 || predicted_index > 2) {
    Serial.println("Invalid prediction index.");
    delay(100);
    return;
  }

  // Phase 4.2: Execute application behavior - print classification with emoji
  Serial.print("RGB norm [R,G,B] = [");
  Serial.print(red, 3); Serial.print(", ");
  Serial.print(green, 3); Serial.print(", ");
  Serial.print(blue, 3); Serial.print("] -> ");
  Serial.print(kLabelNames[predicted_index]);
  Serial.print(" ");
  Serial.println(kLabelEmojis[predicted_index]);

  delay(200);
}