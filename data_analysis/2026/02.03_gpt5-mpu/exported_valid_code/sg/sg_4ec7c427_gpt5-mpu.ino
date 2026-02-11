#include <TensorFlowLite.h>  // Base TFLM header (must precede dependent headers)
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino_APDS9960.h>
#include <Wire.h>
#include <Arduino_HTS221.h>
#include <Arduino_LPS22HB.h>

#include "model.h"  // Model file (required)

/*
  Object Classifier by Color
  - Board: Arduino Nano 33 BLE Sense
  - Sensor: APDS9960 RGB
  - Model:
      input:  [1, 3] float32  -> [Red, Green, Blue] normalized (sum to 1)
      output: [1, 3] uint8    -> 3 classes (Apple, Banana, Orange), quantized
  - Serial: 9600 baud, outputs class with emoji
*/

// Phase 1: Initialization (TFLM core elements)
// 1.2: Declare critical variables
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

static const tflite::Model* tflm_model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// 1.3: Define tensor arena
constexpr int kTensorArenaSize = 16384;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Application specifics
static const char* kClassNames[3] = {
  "Apple 🍎",
  "Banana 🍌",
  "Orange 🍊"
};

// Status flag to avoid running loop when init fails
static bool g_ready = false;

// Utility: Read and normalize RGB from APDS9960 to match dataset distribution (sum to 1)
static bool readNormalizedRGB(float rgb[3]) {
  int r, g, b, c;
  // Wait for fresh data
  uint32_t start_ms = millis();
  while (!APDS.colorAvailable()) {
    if (millis() - start_ms > 100) break;
    delay(5);
  }

  if (!APDS.readColor(r, g, b, c)) {
    return false;
  }

  // Normalize by sum (avoid division by zero)
  float rf = static_cast<float>(r);
  float gf = static_cast<float>(g);
  float bf = static_cast<float>(b);
  float sum = rf + gf + bf;
  if (sum <= 0.0f) {
    float cs = static_cast<float>(c);
    if (cs <= 0.0f) return false;
    rgb[0] = rgb[1] = rgb[2] = 1.0f / 3.0f;
    return true;
  }

  rgb[0] = rf / sum;  // Red
  rgb[1] = gf / sum;  // Green
  rgb[2] = bf / sum;  // Blue

  // Clamp to [0,1]
  for (int i = 0; i < 3; i++) {
    if (rgb[i] < 0.0f) rgb[i] = 0.0f;
    if (rgb[i] > 1.0f) rgb[i] = 1.0f;
  }
  return true;
}

// Pretty-print output scores (handles quantized uint8 dequantization)
static void printScores(const TfLiteTensor* out) {
  if (out->type == kTfLiteUInt8) {
    float s = out->params.scale;
    int zp = out->params.zero_point;
    Serial.print("scores: [");
    for (int i = 0; i < 3; i++) {
      uint8_t q = out->data.uint8[i];
      float f = s * (static_cast<int>(q) - zp);
      Serial.print(f, 4);
      if (i < 2) Serial.print(", ");
    }
    Serial.println("]");
  } else if (out->type == kTfLiteFloat32) {
    Serial.print("scores: [");
    for (int i = 0; i < 3; i++) {
      Serial.print(out->data.f[i], 4);
      if (i < 2) Serial.print(", ");
    }
    Serial.println("]");
  } else {
    Serial.println("scores: [unsupported dtype]");
  }
}

void setup() {
  // Serial setup
  Serial.begin(9600);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color (APDS9960 + TFLM)");
  Serial.println("Initializing...");

  // Phase 2.1: Sensor setup
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor.");
    return;
  }
  Serial.println("APDS9960 initialized.");

  // Phase 1.4: Load the model (use model array from model.h)
  tflm_model = tflite::GetModel(::model);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema mismatch. Model schema: ");
    Serial.print(tflm_model->version());
    Serial.print(" != TFLITE_SCHEMA_VERSION: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  // Phase 1.5: Resolve Operators
  static tflite::AllOpsResolver resolver;

  // Phase 1.6: Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter, nullptr);
  interpreter = &static_interpreter;

  // Phase 1.7: Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    return;
  }

  // Phase 1.8: Define Model Inputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Validate input tensor
  bool input_ok = true;
  if (input->type != kTfLiteFloat32) {
    Serial.println("ERROR: Input tensor must be float32.");
    input_ok = false;
  }
  if (!(input->dims->size == 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.println("ERROR: Input tensor must have shape [1,3].");
    input_ok = false;
  }

  // Validate output tensor
  bool output_ok = true;
  if (!(output->dims->size == 2 && output->dims->data[0] == 1 && output->dims->data[1] == 3)) {
    Serial.println("ERROR: Output tensor must have shape [1,3].");
    output_ok = false;
  }
  if (!(output->type == kTfLiteUInt8 || output->type == kTfLiteFloat32)) {
    Serial.println("ERROR: Output tensor must be uint8 (quantized) or float32.");
    output_ok = false;
  }

  if (!input_ok || !output_ok) {
    return;
  }

  // Phase 1.9: Other relevant parts (none beyond sensor/serial for this app)
  Serial.println("Initialization complete. Starting inference.");
  g_ready = true;
}

void loop() {
  if (!g_ready) {
    delay(500);
    return;
  }

  // Phase 2: Preprocessing - acquire and normalize RGB
  float rgb[3];
  if (!readNormalizedRGB(rgb)) {
    Serial.println("WARN: Failed to read RGB. Retrying...");
    delay(100);
    return;
  }

  // Optional: print normalized input
  Serial.print("input RGB(norm): [");
  Serial.print(rgb[0], 3); Serial.print(", ");
  Serial.print(rgb[1], 3); Serial.print(", ");
  Serial.print(rgb[2], 3); Serial.println("]");

  // Phase 3.1: Copy data to input tensor
  input->data.f[0] = rgb[0];  // Red
  input->data.f[1] = rgb[1];  // Green
  input->data.f[2] = rgb[2];  // Blue

  // Phase 3.2: Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(250);
    return;
  }

  // Phase 4.1: Process output
  int top_index = 0;
  if (output->type == kTfLiteUInt8) {
    uint8_t best_val = output->data.uint8[0];
    for (int i = 1; i < 3; i++) {
      uint8_t v = output->data.uint8[i];
      if (v > best_val) {
        best_val = v;
        top_index = i;
      }
    }
  } else { // kTfLiteFloat32
    float best_val = output->data.f[0];
    for (int i = 1; i < 3; i++) {
      float v = output->data.f[i];
      if (v > best_val) {
        best_val = v;
        top_index = i;
      }
    }
  }

  // Phase 4.2: Execute application behavior - print class with emoji
  Serial.print("Prediction: ");
  Serial.println(kClassNames[top_index]);
  printScores(output);
  Serial.println();

  delay(250);
}