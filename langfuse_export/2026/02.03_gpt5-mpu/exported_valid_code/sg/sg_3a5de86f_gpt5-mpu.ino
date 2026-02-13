#include <Arduino.h>
#include <Wire.h>
#include <Arduino_APDS9960.h>

// Base TFLite Micro header MUST come before other TFLM headers
#include "TensorFlowLite.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "model.h"

// Application constants
static const uint32_t kSerialBaud = 9600;
static const size_t kTensorArenaSize = 8192;

// Class labels and emojis
static const char* kClasses[3] = { "Apple", "Banana", "Orange" };
static const char* kEmojis[3]  = { "🍎",     "🍌",     "🍊"     };

// TFLM globals
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflm_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// Helper to obtain model data pointer from included model.h
const unsigned char* ResolveModelDataPtr() {
  return model;
}

// Optional: translate TfLiteType to string for debug
const char* TypeToStr(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "float32";
    case kTfLiteInt8:    return "int8";
    case kTfLiteUInt8:   return "uint8";
    case kTfLiteInt16:   return "int16";
    case kTfLiteInt32:   return "int32";
    default:             return "other";
  }
}

// Read RGB from APDS9960 and compute normalized features matching dataset semantics
// Returns true if a new color reading was obtained and normalized into feats[3]
bool ReadNormalizedRGB(float feats[3]) {
  if (!APDS.colorAvailable()) {
    return false;
  }
  int r = 0, g = 0, b = 0;
  APDS.readColor(r, g, b);

  // Normalize to fractions that sum to ~1.0 to match dataset stats
  const float rf = (float)max(r, 0);
  const float gf = (float)max(g, 0);
  const float bf = (float)max(b, 0);
  const float sum = rf + gf + bf;
  const float eps = 1e-6f;
  feats[0] = (sum > eps) ? (rf / sum) : 0.0f;  // Red
  feats[1] = (sum > eps) ? (gf / sum) : 0.0f;  // Green
  feats[2] = (sum > eps) ? (bf / sum) : 0.0f;  // Blue
  return true;
}

void setup() {
  // Phase 1: Initialization
  Serial.begin(kSerialBaud);
  while (!Serial) { delay(10); }

  Wire.begin();

  // Initialize sensor
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 color sensor.");
    while (1) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // Set up TFLM error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model from model.h
  const unsigned char* model_data_ptr = ResolveModelDataPtr();
  if (model_data_ptr == nullptr) {
    Serial.println("ERROR: Model data pointer not found. Check model.h symbols.");
    while (1) { delay(1000); }
  }
  tflm_model = tflite::GetModel(model_data_ptr);
  if (tflm_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema mismatch. Model: ");
    Serial.print(tflm_model->version());
    Serial.print(" != Runtime: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  // Resolve operators (fallback to all ops as architecture is unknown)
  static tflite::AllOpsResolver resolver;

  // Instantiate interpreter
  interpreter = new tflite::MicroInterpreter(tflm_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  if (!interpreter) {
    Serial.println("ERROR: Interpreter allocation failed.");
    while (1) { delay(1000); }
  }

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (1) { delay(1000); }
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Input checks
  if (input->type != kTfLiteFloat32) {
    Serial.print("ERROR: Expected input type float32, got ");
    Serial.println(TypeToStr(input->type));
    while (1) { delay(1000); }
  }
  if (!(input->dims && input->dims->size >= 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.println("ERROR: Expected input dims [1,3].");
    while (1) { delay(1000); }
  }

  // Output checks
  if (output->type != kTfLiteUInt8) {
    Serial.print("ERROR: Expected output type uint8, got ");
    Serial.println(TypeToStr(output->type));
    while (1) { delay(1000); }
  }
  if (!(output->dims && output->dims->size >= 2 && output->dims->data[0] == 1 && output->dims->data[1] == 3)) {
    Serial.println("ERROR: Expected output dims [1,3].");
    while (1) { delay(1000); }
  }

  Serial.println("TFLM initialized.");
  Serial.print("Input: type=");
  Serial.print(TypeToStr(input->type));
  Serial.print(" dims=[");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(",");
  }
  Serial.println("]");

  Serial.print("Output: type=");
  Serial.print(TypeToStr(output->type));
  Serial.print(" dims=[");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print(",");
  }
  Serial.println("]");

  Serial.println("Object Classifier by Color is ready.");
}

void loop() {
  // Phase 2: Preprocessing - read and normalize RGB
  float feats[3] = {0, 0, 0};
  if (!ReadNormalizedRGB(feats)) {
    delay(5);
    return;
  }

  // Phase 3: Inference
  // Copy data to input tensor
  input->data.f[0] = feats[0];  // Red
  input->data.f[1] = feats[1];  // Green
  input->data.f[2] = feats[2];  // Blue

  // Invoke
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(50);
    return;
  }

  // Phase 4: Postprocessing
  // Output is uint8 scores for 3 classes
  uint8_t s0 = output->data.uint8[0];
  uint8_t s1 = output->data.uint8[1];
  uint8_t s2 = output->data.uint8[2];

  // Argmax
  int max_idx = 0;
  uint8_t max_val = s0;
  if (s1 > max_val) { max_val = s1; max_idx = 1; }
  if (s2 > max_val) { max_val = s2; max_idx = 2; }

  // Print results
  Serial.print("RGB_norm: R=");
  Serial.print(feats[0], 3);
  Serial.print(" G=");
  Serial.print(feats[1], 3);
  Serial.print(" B=");
  Serial.print(feats[2], 3);
  Serial.print(" | Scores: [");
  Serial.print((float)s0 / 255.0f, 3);
  Serial.print(", ");
  Serial.print((float)s1 / 255.0f, 3);
  Serial.print(", ");
  Serial.print((float)s2 / 255.0f, 3);
  Serial.print("] => Class: ");
  Serial.print(kClasses[max_idx]);
  Serial.print(" ");
  Serial.println(kEmojis[max_idx]);

  delay(150);
}