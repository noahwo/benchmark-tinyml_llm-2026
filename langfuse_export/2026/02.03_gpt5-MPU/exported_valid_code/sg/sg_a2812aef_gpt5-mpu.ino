#include <TensorFlowLite.h>  // Base TFLM library MUST be included before dependent headers
#include "model.h"           // Model flatbuffer header (provided externally)

// TFLM dependent headers
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Sensor library
#include <Arduino_APDS9960.h>

// Application constants
static const int kSerialBaud = 9600;
static const int kNumFeatures = 3;  // Red, Green, Blue
static const int kNumClasses = 3;   // Apple, Banana, Orange

// Labels and emojis for classes
const char* kClassNames[kNumClasses] = {"Apple", "Banana", "Orange"};
const char* kClassEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};

// TFLM globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* tflite_model = nullptr;  // Renamed to avoid conflict with model[] from model.h
tflite::AllOpsResolver resolver;  // Fallback resolver if ops are unknown
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena
constexpr int kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Utility: read and normalize RGB from APDS9960 into [0,1] that sum to 1
bool readNormalizedRGB(float& r_norm, float& g_norm, float& b_norm) {
  // Wait for a color sample to be available
  if (!APDS.colorAvailable()) {
    return false;
  }

  int r = 0, g = 0, b = 0;
  // Arduino_APDS9960::readColor returns void on Nano 33 BLE Sense library
  APDS.readColor(r, g, b);

  // Prevent negatives and compute sum
  if (r < 0) r = 0;
  if (g < 0) g = 0;
  if (b < 0) b = 0;
  const float sum = static_cast<float>(r + g + b);
  if (sum <= 0.0f) {
    r_norm = g_norm = b_norm = 0.0f;
    return true;
  }

  r_norm = static_cast<float>(r) / sum;
  g_norm = static_cast<float>(g) / sum;
  b_norm = static_cast<float>(b) / sum;
  return true;
}

// Utility: print tensor shape and type (for debugging/validation)
void printTensorInfo(const char* name, const TfLiteTensor* t) {
  Serial.print(name);
  Serial.print(" dims=[");
  for (int i = 0; i < t->dims->size; i++) {
    Serial.print(t->dims->data[i]);
    if (i < t->dims->size - 1) Serial.print(", ");
  }
  Serial.print("] type=");
  switch (t->type) {
    case kTfLiteFloat32: Serial.println("FLOAT32"); break;
    case kTfLiteUInt8: Serial.println("UINT8"); break;
    case kTfLiteInt8: Serial.println("INT8"); break;
    default: Serial.println("OTHER"); break;
  }
}

void setup() {
  // Phase 1: Initialization
  Serial.begin(kSerialBaud);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color - TFLite Micro on Arduino Nano 33 BLE Sense");

  // Sensor initialization (Phase 2.1)
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 color sensor.");
    while (1) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // Load model (Phase 1.4)
  // 'model' is the flatbuffer byte array provided by model.h
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch: expected ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(", got ");
    Serial.println(tflite_model->version());
    while (1) { delay(1000); }
  }
  Serial.println("Model loaded.");

  // Instantiate interpreter (Phase 1.6)
  interpreter = new tflite::MicroInterpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Allocate tensors (Phase 1.7)
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Define model inputs (Phase 1.8)
  Serial.println("Input tensor info:");
  printTensorInfo("  input[0]", input);
  Serial.println("Output tensor info:");
  printTensorInfo("  output[0]", output);

  // Validate input expectations: [1,3] float32
  bool input_ok = (input->type == kTfLiteFloat32) &&
                  (input->dims->size == 2) &&
                  (input->dims->data[0] == 1) &&
                  (input->dims->data[1] == kNumFeatures);
  if (!input_ok) {
    Serial.println("ERROR: Unexpected input tensor specification. Expected [1,3] float32.");
    while (1) { delay(1000); }
  }

  Serial.println("Setup complete.");
}

void loop() {
  // Phase 2: Preprocessing - get sensor data and normalize
  float r = 0.0f, g = 0.0f, b = 0.0f;
  if (!readNormalizedRGB(r, g, b)) {
    delay(5);
    return;
  }

  // Phase 3.1: Copy data to input tensor
  input->data.f[0] = r;
  input->data.f[1] = g;
  input->data.f[2] = b;

  // Phase 3.2: Invoke interpreter
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed");
    delay(50);
    return;
  }

  // Phase 4.1: Process output
  int best_idx = 0;
  float best_score = -1.0f;
  float scores[kNumClasses];

  if (output->type == kTfLiteUInt8) {
    // Quantized uint8 outputs assumed to be 0..255
    for (int i = 0; i < kNumClasses; i++) {
      uint8_t v = output->data.uint8[i];
      scores[i] = static_cast<float>(v) / 255.0f;
      if (scores[i] > best_score) {
        best_score = scores[i];
        best_idx = i;
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < kNumClasses; i++) {
      float v = output->data.f[i];
      scores[i] = v;
      if (v > best_score) {
        best_score = v;
        best_idx = i;
      }
    }
  } else if (output->type == kTfLiteInt8) {
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    for (int i = 0; i < kNumClasses; i++) {
      int8_t q = output->data.int8[i];
      float v = scale * (static_cast<int>(q) - zero_point);
      scores[i] = v;
      if (v > best_score) {
        best_score = v;
        best_idx = i;
      }
    }
  } else {
    Serial.println("ERROR: Unsupported output tensor type");
    delay(100);
    return;
  }

  // Phase 4.2: Execute application behavior - print result with emoji
  Serial.print("RGB(norm): ");
  Serial.print(r, 3); Serial.print(", ");
  Serial.print(g, 3); Serial.print(", ");
  Serial.print(b, 3);

  Serial.print("  ->  Pred: ");
  Serial.print(kClassNames[best_idx]);
  Serial.print(" ");
  Serial.print(kClassEmojis[best_idx]);
  Serial.print("  (scores: ");

  for (int i = 0; i < kNumClasses; i++) {
    Serial.print(kClassNames[i]);
    Serial.print("=");
    Serial.print(scores[i], 3);
    if (i < kNumClasses - 1) Serial.print(", ");
  }
  Serial.println(")");

  delay(150);
}