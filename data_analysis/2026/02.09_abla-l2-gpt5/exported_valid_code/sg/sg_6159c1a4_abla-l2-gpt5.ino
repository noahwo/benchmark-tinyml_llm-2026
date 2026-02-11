#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"  // Must define: const unsigned char model[] = { ... };

//
// Project: Color Object Classifier (Arduino Nano 33 BLE Sense + APDS-9960 + TFLM)
//

// ---- Configuration ----
static const int kTensorArenaSize = 16384;  // Per specification
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static const char* kLabels[3] = {"Apple", "Banana", "Orange"};
static const char* kEmojis[3] = {"🍎", "🍌", "🍊"};

// ---- TFLM globals ----
tflite::ErrorReporter* g_error_reporter = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
const tflite::Model* g_tflite_model = nullptr;
tflite::AllOpsResolver g_resolver;

TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

// ---- Helpers ----
static void printTensorInfo(const TfLiteTensor* t, const char* name) {
  if (!t) return;
  Serial.print(name);
  Serial.print(" type=");
  Serial.print(t->type);
  Serial.print(" dims=[");
  for (int i = 0; i < t->dims->size; i++) {
    Serial.print(t->dims->data[i]);
    if (i < t->dims->size - 1) Serial.print(",");
  }
  Serial.println("]");
}

static bool readNormalizedRGB(float& r, float& g, float& b) {
  // Wait until a color sample is ready
  if (!APDS.colorAvailable()) {
    return false;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  int a = 0;  // ambient (unused)
  APDS.readColor(r_raw, g_raw, b_raw, a);

  // Normalize to sum=1.0 to match dataset distribution (min_max_clip in [0,1])
  const float rf = static_cast<float>(r_raw);
  const float gf = static_cast<float>(g_raw);
  const float bf = static_cast<float>(b_raw);
  const float sum = rf + gf + bf;

  if (sum <= 0.0f) {
    return false;
  }

  r = rf / sum;
  g = gf / sum;
  b = bf / sum;

  // Clip to [0,1] just in case
  r = r < 0.f ? 0.f : (r > 1.f ? 1.f : r);
  g = g < 0.f ? 0.f : (g > 1.f ? 1.f : g);
  b = b < 0.f ? 0.f : (b > 1.f ? 1.f : b);
  return true;
}

static int argmax_uint8(const uint8_t* data, int n) {
  int best_i = 0;
  uint8_t best_v = data[0];
  for (int i = 1; i < n; i++) {
    if (data[i] > best_v) {
      best_v = data[i];
      best_i = i;
    }
  }
  return best_i;
}

static int argmax_float(const float* data, int n) {
  int best_i = 0;
  float best_v = data[0];
  for (int i = 1; i < n; i++) {
    if (data[i] > best_v) {
      best_v = data[i];
      best_i = i;
    }
  }
  return best_i;
}

// ---- Arduino lifecycle ----
void setup() {
  Serial.begin(9600);
  while (!Serial && millis() < 4000) {
    // Wait for Serial to be ready (up to ~4s) to avoid blocking in headless mode
  }
  Serial.println("Color Object Classifier starting...");

  // Sensor init
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS-9960 sensor.");
    while (true) { delay(1000); }
  }
  Serial.println("APDS-9960 initialized.");

  // TFLM Error Reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  g_error_reporter = &micro_error_reporter;

  // Load model from model.h (binary array named 'model')
  g_tflite_model = tflite::GetModel(model);
  if (g_tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema ");
    Serial.print(g_tflite_model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }
  Serial.println("Model loaded.");

  // Create MicroInterpreter
  static tflite::MicroInterpreter static_interpreter(
      g_tflite_model, g_resolver, tensor_arena, kTensorArenaSize, g_error_reporter);
  g_interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus alloc_status = g_interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    while (true) { delay(1000); }
  }
  Serial.println("Tensors allocated.");

  // Get input/output
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  // Optional: print tensor info
  printTensorInfo(g_input, "Input");
  printTensorInfo(g_output, "Output");

  Serial.println("Setup complete.");
}

void loop() {
  // Read and preprocess sensor data
  float r, g, b;
  if (!readNormalizedRGB(r, g, b)) {
    delay(5);  // No data yet; yield a bit
    return;
  }

  // Copy to model input: expected float32 [1,3] order: Red, Green, Blue
  if (g_input->type == kTfLiteFloat32) {
    float* in = g_input->data.f;
    in[0] = r;  // Red
    in[1] = g;  // Green
    in[2] = b;  // Blue
  } else if (g_input->type == kTfLiteUInt8) {
    // Fallback if the model has quantized input; scale 0..1 -> 0..255
    uint8_t* in = g_input->data.uint8;
    in[0] = (uint8_t)(r * 255.0f + 0.5f);
    in[1] = (uint8_t)(g * 255.0f + 0.5f);
    in[2] = (uint8_t)(b * 255.0f + 0.5f);
  } else {
    Serial.print("ERROR: Unsupported input type: ");
    Serial.println(g_input->type);
    delay(500);
    return;
  }

  // Inference
  if (g_interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(200);
    return;
  }

  // Postprocessing: argmax over 3 classes
  int predicted = -1;
  if (g_output->type == kTfLiteUInt8) {
    const uint8_t* out = g_output->data.uint8;
    predicted = argmax_uint8(out, 3);
  } else if (g_output->type == kTfLiteFloat32) {
    const float* out = g_output->data.f;
    predicted = argmax_float(out, 3);
  } else {
    Serial.print("ERROR: Unsupported output type: ");
    Serial.println(g_output->type);
    delay(200);
    return;
  }

  // Bound check and print result
  if (predicted < 0 || predicted > 2) {
    Serial.println("ERROR: Invalid prediction index.");
  } else {
    Serial.print("RGB(norm) = [");
    Serial.print(r, 3); Serial.print(", ");
    Serial.print(g, 3); Serial.print(", ");
    Serial.print(b, 3); Serial.print("]  ->  ");
    Serial.print(kLabels[predicted]);
    Serial.print(" ");
    Serial.println(kEmojis[predicted]);
  }

  delay(250);
}