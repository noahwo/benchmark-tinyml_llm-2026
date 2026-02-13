#include <Arduino.h>
#include <TensorFlowLite.h>  // Base TFLM header (must be included before dependent headers)
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include <Arduino_APDS9960.h>

#include "model.h"  // Contains the TFLite model flatbuffer bytes: const unsigned char model[]

// Application constants
static const uint32_t kBaudRate = 9600;
static const int kTensorArenaSize = 10 * 1024;  // Per specification

// TinyML globals
static tflite::ErrorReporter* error_reporter = nullptr;
static const tflite::Model* tfl_model = nullptr;  // Renamed to avoid conflict with model[] from model.h
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;
static uint8_t tensor_arena[kTensorArenaSize];

// Classification labels and emojis
static const char* kClassNames[3] = { "Apple", "Banana", "Orange" };
static const char* kClassEmojis[3] = { "🍎", "🍌", "🍊" };

// Utility: argmax for 3-element uint8 array
static int argmax_u8_3(const uint8_t* v) {
  int idx = 0;
  uint8_t best = v[0];
  for (int i = 1; i < 3; i++) {
    if (v[i] > best) {
      best = v[i];
      idx = i;
    }
  }
  return idx;
}

// Initialize APDS9960 color sensor
static bool initColorSensor() {
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 color sensor.");
    return false;
  }
  // Give sensor some time to stabilize
  delay(10);
  return true;
}

// Read normalized RGB as fractions summing to ~1.0, based on r/(r+g+b)
static bool readNormalizedRGB(float& r_n, float& g_n, float& b_n) {
  if (!APDS.colorAvailable()) {
    return false;
  }
  int r = 0, g = 0, b = 0;
  APDS.readColor(r, g, b);

  long sum = (long)r + (long)g + (long)b;
  if (sum <= 0) {
    return false;
  }

  r_n = (float)r / (float)sum;
  g_n = (float)g / (float)sum;
  b_n = (float)b / (float)sum;
  return true;
}

void setup() {
  // Phase 1.1: Communication init
  Serial.begin(kBaudRate);
  while (!Serial) { delay(5); }

  Serial.println("Object Classifier by Color - Nano 33 BLE Sense");
  Serial.flush();

  // Phase 1.2 and 1.3: TFLM core setup and tensor arena
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Phase 1.4: Load the model
  tfl_model = tflite::GetModel(model);  // Use byte array 'model' from model.h
  // Validate model if schema macro available; otherwise skip strict check.
  #ifdef TFLITE_SCHEMA_VERSION
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("ERROR: Model schema %d not equal to supported %d.", tfl_model->version(), TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }
  #endif

  // Phase 1.5: Resolve operators (use AllOpsResolver as fallback)
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("ERROR: AllocateTensors() failed.");
    while (1) { delay(1000); }
  }

  // Phase 1.8: Define/verify model inputs and outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input->type != kTfLiteFloat32) {
    error_reporter->Report("ERROR: Input tensor type is not float32.");
    while (1) { delay(1000); }
  }
  if (output->type != kTfLiteUInt8) {
    error_reporter->Report("ERROR: Output tensor type is not uint8.");
    while (1) { delay(1000); }
  }

  // Phase 1.9 and 2.1: Initialize sensor
  if (!initColorSensor()) {
    while (1) { delay(1000); }
  }

  Serial.println("Initialization complete. Waiting for color data...");
  Serial.flush();
}

void loop() {
  // Phase 2: Preprocessing - read and normalize RGB
  float r_n = 0.0f, g_n = 0.0f, b_n = 0.0f;
  if (!readNormalizedRGB(r_n, g_n, b_n)) {
    // No new color data yet
    delay(5);
    return;
  }

  // Clip to [0,1] just in case
  if (r_n < 0.0f) r_n = 0.0f; if (r_n > 1.0f) r_n = 1.0f;
  if (g_n < 0.0f) g_n = 0.0f; if (g_n > 1.0f) g_n = 1.0f;
  if (b_n < 0.0f) b_n = 0.0f; if (b_n > 1.0f) b_n = 1.0f;

  // Phase 3.1: Copy data into input tensor [1,3] float32
  input->data.f[0] = r_n;
  input->data.f[1] = g_n;
  input->data.f[2] = b_n;

  // Phase 3.2: Invoke interpreter
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    error_reporter->Report("ERROR: Inference failed.");
    delay(50);
    return;
  }

  // Phase 4.1: Process output (uint8 logits or scores)
  const uint8_t* scores = output->data.uint8;
  int predicted = argmax_u8_3(scores);

  // Phase 4.2: Execute application behavior - print classification with emoji
  Serial.print("Input RGB(norm): R=");
  Serial.print(r_n, 3);
  Serial.print(" G=");
  Serial.print(g_n, 3);
  Serial.print(" B=");
  Serial.print(b_n, 3);

  Serial.print(" | Scores(u8): [");
  Serial.print(scores[0]); Serial.print(", ");
  Serial.print(scores[1]); Serial.print(", ");
  Serial.print(scores[2]); Serial.print("]");

  Serial.print(" => Class: ");
  Serial.print(kClassNames[predicted]);
  Serial.print(" ");
  Serial.println(kClassEmojis[predicted]);

  Serial.flush();
  delay(75);  // Small delay to avoid flooding the serial monitor
}