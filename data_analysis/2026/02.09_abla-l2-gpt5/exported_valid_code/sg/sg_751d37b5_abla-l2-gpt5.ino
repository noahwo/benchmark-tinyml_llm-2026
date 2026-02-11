#include <Arduino.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include "model.h"

// TensorFlow Lite Micro headers
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// ---------------------------
// Application configuration
// ---------------------------
static const uint32_t kBaudRate = 9600;
static const uint16_t kSampleIntervalMs = 200;     // Sampling interval
static const int kNumChannels = 3;                 // [Red, Green, Blue]
static const int kNumClasses = 3;                  // ["Apple", "Banana", "Orange"]
static const size_t kTensorArenaSize = 16384;

// Class labels and emojis
static const char* kClassLabels[kNumClasses] = { "Apple", "Banana", "Orange" };
static const char* kClassEmojis[kNumClasses] = { "🍎", "🍌", "🍊" };

// ---------------------------
// TFLM globals
// ---------------------------
tflite::ErrorReporter* error_reporter = nullptr;
static tflite::MicroErrorReporter micro_error_reporter;
const tflite::Model* tfl_model = nullptr;  // Note: different name than byte array "model" from model.h

// Reserve op resolver globally to outlive the interpreter
static tflite::MicroMutableOpResolver<10> micro_op_resolver;

// Tensor arena
#if defined(__ARMCC_VERSION)
  __attribute__((aligned(16)))
#elif defined(__GNUC__)
  __attribute__((aligned(16)))
#else
  #pragma message("No alignment attributes available for this compiler.")
#endif
static uint8_t tensor_arena[kTensorArenaSize];

// Interpreter pointer (backed by a static object created in setup)
static tflite::MicroInterpreter* interpreter = nullptr;

// Cached I/O tensor pointers
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// ---------------------------
// Quantization helpers
// ---------------------------
static inline uint8_t quantize_float_to_uint8(float value, float scale, int zero_point) {
  // Clamp to [0, 255] after quantization
  int32_t q = static_cast<int32_t>(roundf(value / scale) + zero_point);
  if (q < 0) q = 0;
  if (q > 255) q = 255;
  return static_cast<uint8_t>(q);
}

static inline float dequantize_uint8_to_float(uint8_t value, float scale, int zero_point) {
  return (static_cast<int>(value) - zero_point) * scale;
}

// ---------------------------
// Utility: print tensor info (optional debug)
// ---------------------------
static void print_tensor_info(const TfLiteTensor* t, const char* name) {
  Serial.print(F("[TENSOR] "));
  Serial.print(name);
  Serial.print(F(" type="));
  switch (t->type) {
    case kTfLiteFloat32: Serial.print(F("float32")); break;
    case kTfLiteUInt8:   Serial.print(F("uint8"));   break;
    case kTfLiteInt8:    Serial.print(F("int8"));    break;
    default:             Serial.print(F("other"));   break;
  }
  Serial.print(F(" shape=["));
  for (int i = 0; i < t->dims->size; ++i) {
    Serial.print(t->dims->data[i]);
    if (i < t->dims->size - 1) Serial.print(',');
  }
  Serial.println(F("]"));
}

// ---------------------------
// Setup
// ---------------------------
void setup() {
  Serial.begin(kBaudRate);
  while (!Serial) { delay(10); }

  Serial.println(F("Object Classifier by Color (Apple 🍎 / Banana 🍌 / Orange 🍊)"));
  Serial.println(F("Initializing sensors..."));

  if (!APDS.begin()) {
    Serial.println(F("ERROR: Failed to initialize APDS-9960 sensor."));
    while (true) { delay(1000); }
  }
  Serial.println(F("APDS-9960 ready."));

  // Set up error reporter
  error_reporter = &micro_error_reporter;

  // Load model from model.h
  // "model" here refers to the byte array defined in model.h
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(F("ERROR: Model schema "));
    Serial.print(tfl_model->version());
    Serial.print(F(" not equal to supported version "));
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }

  // Configure operator resolver with commonly used ops for small classifiers
  // Adjust as needed if your model uses different ops.
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddRelu6();
  micro_op_resolver.AddLogistic();

  // Instantiate the interpreter using a static lifetime object
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println(F("ERROR: AllocateTensors() failed"));
    while (true) { delay(1000); }
  }

  // Cache I/O tensors
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Optional: print tensor information
  print_tensor_info(input_tensor, "input[0]");
  print_tensor_info(output_tensor, "output[0]");

  Serial.println(F("Initialization complete."));
  Serial.println(F("Reading color data and running inference every 200 ms..."));
}

// ---------------------------
// Main loop
// ---------------------------
void loop() {
  static uint32_t last_sample_ms = 0;
  const uint32_t now = millis();
  if (now - last_sample_ms < kSampleIntervalMs) {
    delay(5);
    return;
  }
  last_sample_ms = now;

  // 1) Sensor read
  int r = 0, g = 0, b = 0, c = 0;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, c);
  } else {
    // If not available yet, skip this cycle
    return;
  }

  // 2) Preprocessing: normalize to sum=1.0 -> [Rn, Gn, Bn]
  const int sum_rgb = r + g + b;
  float rn = 0.0f, gn = 0.0f, bn = 0.0f;
  if (sum_rgb > 0) {
    rn = (float)r / (float)sum_rgb;
    gn = (float)g / (float)sum_rgb;
    bn = (float)b / (float)sum_rgb;
    // Clamp defensively
    if (rn < 0) rn = 0; if (rn > 1) rn = 1;
    if (gn < 0) gn = 0; if (gn > 1) gn = 1;
    if (bn < 0) bn = 0; if (bn > 1) bn = 1;
  } else {
    // Avoid division by zero; keep zeros
    rn = gn = bn = 0.0f;
  }

  // 3) Prepare input tensor
  if (input_tensor->type == kTfLiteFloat32) {
    float* in = input_tensor->data.f;
    // Input order: ["Red", "Green", "Blue"]
    in[0] = rn;
    in[1] = gn;
    in[2] = bn;
  } else if (input_tensor->type == kTfLiteUInt8) {
    // Quantize normalized floats into uint8 using input tensor's quant params
    const float scale = input_tensor->params.scale;
    const int zero_point = input_tensor->params.zero_point;
    uint8_t* in = input_tensor->data.uint8;
    in[0] = quantize_float_to_uint8(rn, scale, zero_point);
    in[1] = quantize_float_to_uint8(gn, scale, zero_point);
    in[2] = quantize_float_to_uint8(bn, scale, zero_point);
  } else {
    Serial.println(F("ERROR: Unsupported input tensor type."));
    return;
  }

  // 4) Inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println(F("ERROR: Inference failed."));
    return;
  }

  // 5) Postprocessing: read output, argmax
  int argmax = 0;
  float best_score = -1e9f;

  if (output_tensor->type == kTfLiteUInt8) {
    const uint8_t* scores = output_tensor->data.uint8;
    // Determine last dimension as class count
    int classes = (output_tensor->dims->size > 0)
                  ? output_tensor->dims->data[output_tensor->dims->size - 1]
                  : kNumClasses;
    if (classes <= 0 || classes > kNumClasses) classes = kNumClasses;

    // Dequantize for comparison and user-friendly display
    const float scale = output_tensor->params.scale;
    const int zero_point = output_tensor->params.zero_point;
    for (int i = 0; i < classes; ++i) {
      float s = dequantize_uint8_to_float(scores[i], scale, zero_point);
      if (s > best_score) {
        best_score = s;
        argmax = i;
      }
    }
  } else if (output_tensor->type == kTfLiteFloat32) {
    const float* scores = output_tensor->data.f;
    int classes = (output_tensor->dims->size > 0)
                  ? output_tensor->dims->data[output_tensor->dims->size - 1]
                  : kNumClasses;
    if (classes <= 0 || classes > kNumClasses) classes = kNumClasses;

    for (int i = 0; i < classes; ++i) {
      float s = scores[i];
      if (s > best_score) {
        best_score = s;
        argmax = i;
      }
    }
  } else {
    Serial.println(F("ERROR: Unsupported output tensor type."));
    return;
  }

  // 6) Output results
  Serial.print(F("RGB("));
  Serial.print(r); Serial.print(',');
  Serial.print(g); Serial.print(',');
  Serial.print(b); Serial.print(") C=");
  Serial.print(c);
  Serial.print(F(" | Norm=["));
  Serial.print(rn, 3); Serial.print(',');
  Serial.print(gn, 3); Serial.print(',');
  Serial.print(bn, 3); Serial.print("] -> ");

  Serial.print(kClassLabels[argmax]);
  Serial.print(' ');
  Serial.print(kClassEmojis[argmax]);
  Serial.print(F(" | score="));
  Serial.println(best_score, 4);
}