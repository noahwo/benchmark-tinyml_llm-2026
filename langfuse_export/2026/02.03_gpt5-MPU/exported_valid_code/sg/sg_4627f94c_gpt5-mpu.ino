#include <TensorFlowLite.h>  // Base TFLite Micro header must be included first
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/version.h"

#include <Wire.h>
#include <Arduino_APDS9960.h>
#include <Arduino_HTS221.h>
#include <Arduino_LPS22HB.h>

#include "model.h"

// Tensor arena size from application specifications
constexpr int kTensorArenaSize = 8192;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// TFLite Micro globals
static tflite::ErrorReporter* error_reporter = nullptr;
static const tflite::Model* tflite_model = nullptr;
static tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// Application specifics
static const char* kClasses[3] = { "Apple", "Banana", "Orange" };
static const char* kEmojis[3]  = { "🍎", "🍌", "🍊" };

// Forward declarations
void printTensorInfo(const TfLiteTensor* t, const char* name);
int argmax_uint8(const uint8_t* data, int len);
void printProbabilitiesUint8(const uint8_t* data, int len);

// Optional alternative symbol names if provided by model.h (unused if absent)
extern const unsigned char g_model[];     // model data (flatbuffer)
extern const int g_model_len;             // model length

void setup() {
  // Phase 1: Initialization
  Serial.begin(9600);
  unsigned long startWait = millis();
  while (!Serial && (millis() - startWait) < 2000) {
    delay(10);
  }

  Wire.begin();

  // Initialize optional onboard sensors (not strictly required, but included as per spec)
  // HTS and LPS initializations are safe to ignore failure for this application.
  HTS.begin();
  BARO.begin();

  if (!APDS.begin()) {
    Serial.println("Error: Failed to initialize APDS9960 color sensor.");
  } else {
    Serial.println("APDS9960 initialized.");
  }

  // 1.2 Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // 1.4 Load the model (prefer symbol 'model' from model.h if present)
  // model.h in this project defines: const unsigned char model[] = {...};
  tflite_model = tflite::GetModel(model);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema %d not equal to supported %d.", tflite_model->version(), TFLITE_SCHEMA_VERSION);
    Serial.println("Error: Model schema version mismatch.");
    // Continue, but inference may fail
  }

  // 1.5/1.6 Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // 1.7 Allocate memory
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("Error: AllocateTensors() failed.");
    while (true) {
      delay(1000);
    }
  }

  // 1.8 Define Model Inputs/Outputs
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model input tensor:");
  printTensorInfo(input, "input");
  Serial.println("Model output tensor:");
  printTensorInfo(output, "output");

  // Validate input tensor: [1, 3], float32
  bool input_ok = true;
  if (!(input->type == kTfLiteFloat32)) {
    Serial.println("Error: Input tensor type is not float32.");
    input_ok = false;
  }
  if (!(input->dims->size == 2 && input->dims->data[0] == 1 && input->dims->data[1] == 3)) {
    Serial.println("Error: Input tensor shape is not [1, 3].");
    input_ok = false;
  }

  // Validate output tensor: [1, 3], uint8
  bool output_ok = true;
  if (!(output->type == kTfLiteUInt8)) {
    Serial.println("Error: Output tensor type is not uint8.");
    output_ok = false;
  }
  if (!(output->dims->size == 2 && output->dims->data[0] == 1 && output->dims->data[1] == 3)) {
    Serial.println("Error: Output tensor shape is not [1, 3].");
    output_ok = false;
  }

  if (!input_ok || !output_ok) {
    Serial.println("Tensor shape/type validation failed. Check model.h or conversion settings.");
  }

  Serial.println("Setup complete. Starting inference loop...");
}

void loop() {
  // Phase 2: Preprocessing - Sensor Setup and Data Acquisition
  int r = 0, g = 0, b = 0, a = 0;

  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, a);
  } else {
    // If color not available, short delay and try again
    delay(10);
    return;
  }

  // Normalize RGB to sum-to-one as per dataset characteristics
  float rf = static_cast<float>(r);
  float gf = static_cast<float>(g);
  float bf = static_cast<float>(b);
  float sum = rf + gf + bf;

  if (sum <= 0.0f) {
    // Avoid division by zero
    delay(10);
    return;
  }

  float rn = rf / sum;
  float gn = gf / sum;
  float bn = bf / sum;

  // Clamp to [0,1] for safety
  rn = rn < 0.f ? 0.f : (rn > 1.f ? 1.f : rn);
  gn = gn < 0.f ? 0.f : (gn > 1.f ? 1.f : gn);
  bn = bn < 0.f ? 0.f : (bn > 1.f ? 1.f : bn);

  // Phase 3: Inference
  // 3.1 Copy data into input tensor
  if (input && input->type == kTfLiteFloat32 && input->dims->size == 2 && input->dims->data[1] == 3) {
    input->data.f[0] = rn;
    input->data.f[1] = gn;
    input->data.f[2] = bn;
  } else {
    Serial.println("Error: Input tensor not prepared correctly.");
    delay(100);
    return;
  }

  // 3.2 Invoke interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error: Inference invocation failed.");
    delay(100);
    return;
  }

  // Phase 4: Postprocessing
  if (!(output && output->type == kTfLiteUInt8 && output->dims->size == 2 && output->dims->data[1] == 3)) {
    Serial.println("Error: Output tensor not prepared correctly.");
    delay(100);
    return;
  }

  const uint8_t* out_data = output->data.uint8;
  int pred_idx = argmax_uint8(out_data, 3);

  // Print results
  Serial.print("Raw RGB: ");
  Serial.print(r); Serial.print(", ");
  Serial.print(g); Serial.print(", ");
  Serial.print(b);
  Serial.print(" | Norm: ");
  Serial.print(rn, 3); Serial.print(", ");
  Serial.print(gn, 3); Serial.print(", ");
  Serial.print(bn, 3);

  Serial.print(" | Pred: ");
  Serial.print(kClasses[pred_idx]);
  Serial.print(" ");
  Serial.print(kEmojis[pred_idx]);
  Serial.print(" | P=[");
  // Probabilities (uint8 scaled 0..255)
  Serial.print((float)out_data[0] / 255.0f, 2); Serial.print(", ");
  Serial.print((float)out_data[1] / 255.0f, 2); Serial.print(", ");
  Serial.print((float)out_data[2] / 255.0f, 2); Serial.println("]");

  delay(150);
}

void printTensorInfo(const TfLiteTensor* t, const char* name) {
  if (!t) {
    Serial.print(name);
    Serial.println(": (null)");
    return;
  }
  Serial.print(name);
  Serial.print(" type=");
  switch (t->type) {
    case kTfLiteFloat32: Serial.print("float32"); break;
    case kTfLiteUInt8: Serial.print("uint8"); break;
    case kTfLiteInt8: Serial.print("int8"); break;
    case kTfLiteInt16: Serial.print("int16"); break;
    default: Serial.print("other"); break;
  }
  Serial.print(" shape=[");
  if (t->dims) {
    for (int i = 0; i < t->dims->size; i++) {
      Serial.print(t->dims->data[i]);
      if (i < t->dims->size - 1) Serial.print(", ");
    }
  }
  Serial.println("]");
}

int argmax_uint8(const uint8_t* data, int len) {
  if (!data || len <= 0) return 0;
  int idx = 0;
  uint8_t best = data[0];
  for (int i = 1; i < len; i++) {
    if (data[i] > best) {
      best = data[i];
      idx = i;
    }
  }
  return idx;
}

void printProbabilitiesUint8(const uint8_t* data, int len) {
  Serial.print("[");
  for (int i = 0; i < len; i++) {
    float p = (float)data[i] / 255.0f;
    Serial.print(p, 2);
    if (i < len - 1) Serial.print(", ");
  }
  Serial.println("]");
}