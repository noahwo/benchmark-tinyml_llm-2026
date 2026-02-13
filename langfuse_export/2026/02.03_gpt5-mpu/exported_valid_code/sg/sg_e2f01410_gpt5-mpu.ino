#include <Arduino.h>
#include <TensorFlowLite.h>  // Base TFLM header must come before micro/* headers
#include "model.h"           // Model flatbuffer bytes
#include <Arduino_APDS9960.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Application configuration
static const uint32_t kSerialBaud = 9600;
static const int kArenaSize = 16384; // per spec
static const int kNumClasses = 3;
static const char* kClassNames[kNumClasses] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[kNumClasses] = {"🍎", "🍌", "🍊"};

// TFLM globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

alignas(16) static uint8_t tensor_arena[kArenaSize];

// Utility: read and normalize RGB to chromaticity (r+g+b=1)
bool readNormalizedRGB(float& r_n, float& g_n, float& b_n, int num_samples = 5) {
  long r_acc = 0;
  long g_acc = 0;
  long b_acc = 0;
  int samples = 0;

  unsigned long startWait = millis();
  while (!APDS.colorAvailable()) {
    if (millis() - startWait > 100) break; // avoid long blocking
    delay(5);
  }

  for (int i = 0; i < num_samples; ++i) {
    if (!APDS.colorAvailable()) {
      delay(5);
      continue;
    }
    int r = 0, g = 0, b = 0;
    APDS.readColor(r, g, b);
    r_acc += r;
    g_acc += g;
    b_acc += b;
    samples++;
    delay(5);
  }

  if (samples == 0) return false;

  float r = static_cast<float>(r_acc) / samples;
  float g = static_cast<float>(g_acc) / samples;
  float b = static_cast<float>(b_acc) / samples;
  float sum = r + g + b;
  if (sum <= 0.0f) return false;

  r_n = r / sum;
  g_n = g / sum;
  b_n = b / sum;
  return true;
}

void setup() {
  Serial.begin(kSerialBaud);
  while (!Serial) { delay(10); }

  Serial.println("Object Classifier by Color - TinyML (Nano 33 BLE Sense)");

  // Sensor init
  if (!APDS.begin()) {
    Serial.println("Error: Failed to initialize APDS9960 color sensor.");
  } else {
    Serial.println("APDS9960 initialized.");
  }

  // TFLM setup
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model from model.h (must provide model bytes)
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema mismatch. Found: ");
    Serial.print(tflite_model->version());
    Serial.print(" Expected: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Resolver (use AllOpsResolver as architecture is unknown)
  static tflite::AllOpsResolver resolver;

  // Interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) { delay(1000); }
  }

  // Input validation
  input = interpreter->input(0);
  if (input == nullptr) {
    Serial.println("Error: input tensor is null.");
    while (true) { delay(1000); }
  }

  // Expect float32 input of length 3
  if (input->type != kTfLiteFloat32) {
    Serial.println("Error: Model input is not float32 as specified.");
    while (true) { delay(1000); }
  }

  int input_elems = 1;
  for (int i = 0; i < input->dims->size; ++i) {
    input_elems *= input->dims->data[i];
  }
  if (input_elems != 3) {
    Serial.print("Error: Model input size mismatch. Expected 3, got ");
    Serial.println(input_elems);
    while (true) { delay(1000); }
  }

  // Output info (informational)
  TfLiteTensor* output = interpreter->output(0);
  if (output == nullptr) {
    Serial.println("Error: output tensor is null.");
    while (true) { delay(1000); }
  }
  int output_elems = 1;
  for (int i = 0; i < output->dims->size; ++i) {
    output_elems *= output->dims->data[i];
  }
  Serial.print("Output elements: ");
  Serial.println(output_elems);
  Serial.print("Output type: ");
  Serial.println(output->type == kTfLiteUInt8 ? "uint8" :
                 output->type == kTfLiteInt8  ? "int8"  :
                 output->type == kTfLiteFloat32 ? "float32" : "other");

  Serial.println("Setup complete.");
}

void loop() {
  // Phase 2: Preprocessing - read and normalize RGB
  float r_n = 0, g_n = 0, b_n = 0;
  if (!readNormalizedRGB(r_n, g_n, b_n)) {
    Serial.println("Warning: Failed to read color. Retrying...");
    delay(50);
    return;
  }

  // Phase 3.1: Copy data to model input
  input->data.f[0] = r_n;
  input->data.f[1] = g_n;
  input->data.f[2] = b_n;

  // Phase 3.2: Invoke interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error: Inference failed.");
    delay(100);
    return;
  }

  // Phase 4: Postprocessing
  TfLiteTensor* output = interpreter->output(0);

  int top_idx = 0;
  float top_score = -1e30f;

  if (output->type == kTfLiteUInt8) {
    // Argmax on quantized outputs (affine transform preserves ordering)
    const uint8_t* out = output->data.uint8;
    int count = 1;
    for (int i = 0; i < output->dims->size; ++i) count *= output->dims->data[i];

    for (int i = 0; i < count; ++i) {
      float score = static_cast<float>(out[i]); // scale/zero_point not needed for argmax
      if (score > top_score) {
        top_score = score;
        top_idx = i;
      }
    }
  } else if (output->type == kTfLiteFloat32) {
    const float* out = output->data.f;
    int count = 1;
    for (int i = 0; i < output->dims->size; ++i) count *= output->dims->data[i];

    for (int i = 0; i < count; ++i) {
      float score = out[i];
      if (score > top_score) {
        top_score = score;
        top_idx = i;
      }
    }
  } else if (output->type == kTfLiteInt8) {
    const int8_t* out = output->data.int8;
    int count = 1;
    for (int i = 0; i < output->dims->size; ++i) count *= output->dims->data[i];

    for (int i = 0; i < count; ++i) {
      float score = static_cast<float>(out[i]); // argmax only
      if (score > top_score) {
        top_score = score;
        top_idx = i;
      }
    }
  } else {
    Serial.println("Unsupported output tensor type.");
    delay(200);
    return;
  }

  // Prepare human-readable scores (optional dequantization for display)
  float scores[kNumClasses] = {0};
  int count = 1;
  for (int i = 0; i < output->dims->size; ++i) count *= output->dims->data[i];
  count = min(count, kNumClasses);

  if (output->type == kTfLiteUInt8) {
    float scale = output->params.scale;
    int zp = output->params.zero_point;
    for (int i = 0; i < count; ++i) {
      scores[i] = scale * (static_cast<int>(output->data.uint8[i]) - zp);
    }
  } else if (output->type == kTfLiteInt8) {
    float scale = output->params.scale;
    int zp = output->params.zero_point;
    for (int i = 0; i < count; ++i) {
      scores[i] = scale * (static_cast<int>(output->data.int8[i]) - zp);
    }
  } else if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < count; ++i) {
      scores[i] = output->data.f[i];
    }
  }

  // Serial output: emoji + label + scores + input RGB
  Serial.print(kClassEmojis[top_idx]);
  Serial.print(" ");
  Serial.print(kClassNames[top_idx]);
  Serial.print(" | RGB(norm): ");
  Serial.print(r_n, 3); Serial.print(", ");
  Serial.print(g_n, 3); Serial.print(", ");
  Serial.print(b_n, 3);
  Serial.print(" | scores: ");
  for (int i = 0; i < count; ++i) {
    Serial.print(kClassNames[i]);
    Serial.print("=");
    Serial.print(scores[i], 3);
    if (i < count - 1) Serial.print(", ");
  }
  Serial.println();

  delay(200);
}