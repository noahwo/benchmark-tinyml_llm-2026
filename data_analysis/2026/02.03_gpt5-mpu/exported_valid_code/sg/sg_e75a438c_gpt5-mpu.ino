#include <Arduino.h>
#include <Wire.h>

// TensorFlow Lite Micro base header must come before dependent headers
#include <TensorFlowLite.h>
#include "model.h"  // Model binary (model array)

#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <Arduino_APDS9960.h>

// Application metadata
static const char* kAppName = "Object Classifier by Color";

// Classification labels and emojis
static const char* kClassNames[3] = {"Apple", "Banana", "Orange"};
static const char* kClassEmojis[3] = {"🍎", "🍌", "🍊"};

// TFLite Micro globals (keep them static/global to persist across inferences)
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* tfl_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;

  // Tensor arena size per specification
  constexpr int kTensorArenaSize = 16384;
  // Ensure sufficient alignment for the tensor arena
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
}

// Helper: Print tensor shape
void printTensorShape(const TfLiteTensor* t) {
  if (!t || !t->dims) {
    Serial.println("  shape: (null)");
    return;
  }
  Serial.print("  shape: [");
  for (int i = 0; i < t->dims->size; i++) {
    Serial.print(t->dims->data[i]);
    if (i < t->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
}

void setup() {
  // Phase 1: Initialization
  Serial.begin(9600);
  while (!Serial) { delay(10); }  // Wait for serial

  Serial.println();
  Serial.println("=======================================");
  Serial.print("Starting: ");
  Serial.println(kAppName);
  Serial.println("Board: Arduino Nano 33 BLE Sense");
  Serial.println("Sensor: APDS9960 (RGB)");
  Serial.println("=======================================");

  // Initialize the color sensor
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize APDS9960 sensor.");
    Serial.println("Please check wiring and ensure the board is Nano 33 BLE Sense.");
    while (true) { delay(1000); }
  }
  Serial.println("APDS9960 initialized.");

  // 1.2 Error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // 1.4 Load the model (model array comes from model.h)
  tfl_model = tflite::GetModel(model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema ");
    Serial.print(tfl_model->version());
    Serial.print(" not equal to supported schema ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true) { delay(1000); }
  }
  Serial.println("Model loaded successfully.");

  // 1.5 Resolve operators
  static tflite::AllOpsResolver resolver;  // Fallback to all ops resolver

  // 1.6 Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // 1.7 Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed");
    while (true) { delay(1000); }
  }
  Serial.println("Tensors allocated.");

  // 1.8 Define Model Inputs
  input = interpreter->input(0);
  Serial.println("Input tensor:");
  Serial.print("  type: "); Serial.println(input->type);
  printTensorShape(input);

  TfLiteTensor* output = interpreter->output(0);
  Serial.println("Output tensor:");
  Serial.print("  type: "); Serial.println(output->type);
  printTensorShape(output);

  // 1.9 Set Up Other Relevant Parts (already initialized Serial and Sensor)
  Serial.println("Initialization complete. Beginning inference loop...");
  Serial.println();
}

void loop() {
  // Phase 2: Preprocessing - Sensor Setup and Data Acquisition
  if (!APDS.colorAvailable()) {
    delay(5);
    return;
  }

  int r_raw = 0, g_raw = 0, b_raw = 0;
  if (!APDS.readColor(r_raw, g_raw, b_raw)) {
    // Reading failed; skip this cycle
    return;
  }

  // Normalize to match dataset characteristics (Red, Green, Blue sum to ~1.0)
  const float epsilon = 1e-6f;
  float sum = static_cast<float>(r_raw) + static_cast<float>(g_raw) + static_cast<float>(b_raw);
  if (sum < epsilon) {
    // Avoid divide-by-zero; skip if dark/no reading
    return;
  }

  float r_n = static_cast<float>(r_raw) / sum;
  float g_n = static_cast<float>(g_raw) / sum;
  float b_n = static_cast<float>(b_raw) / sum;

  // Phase 3: Inference
  // 3.1 Data Copy into input tensor [1, 3] float32
  if (input->type != kTfLiteFloat32) {
    Serial.println("ERROR: Model input is not float32 as expected.");
    delay(100);
    return;
  }
  // Assign values in the order: [Red, Green, Blue]
  input->data.f[0] = r_n;
  input->data.f[1] = g_n;
  input->data.f[2] = b_n;

  // 3.2 Invoke Interpreter
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    delay(50);
    return;
  }

  // Phase 4: Postprocessing
  TfLiteTensor* output = interpreter->output(0);

  // Determine number of classes from output tensor shape
  int num_classes = 1;
  if (output->dims && output->dims->size > 0) {
    for (int i = 0; i < output->dims->size; ++i) {
      num_classes *= output->dims->data[i];
    }
  } else {
    // Fallback if dims are not present
    num_classes = 3;
  }
  if (num_classes > 3) num_classes = 3; // Constrain to known labels

  int argmax = 0;

  if (output->type == kTfLiteUInt8) {
    // Quantized uint8 outputs
    uint8_t max_val = 0;
    for (int i = 0; i < num_classes; ++i) {
      uint8_t v = output->data.uint8[i];
      if (i == 0 || v > max_val) {
        max_val = v;
        argmax = i;
      }
    }

    // Print results
    Serial.print("Input (norm RGB): R=");
    Serial.print(r_n, 3);
    Serial.print(" G=");
    Serial.print(g_n, 3);
    Serial.print(" B=");
    Serial.print(b_n, 3);
    Serial.print(" | Scores (u8): [");
    for (int i = 0; i < num_classes; ++i) {
      Serial.print(output->data.uint8[i]);
      if (i < num_classes - 1) Serial.print(", ");
    }
    Serial.print("] -> Pred: ");
    Serial.print(kClassNames[argmax]);
    Serial.print(" ");
    Serial.println(kClassEmojis[argmax]);

  } else if (output->type == kTfLiteFloat32) {
    // Float outputs (fallback if model differs)
    float max_val = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
      float v = output->data.f[i];
      if (i == 0 || v > max_val) {
        max_val = v;
        argmax = i;
      }
    }

    Serial.print("Input (norm RGB): R=");
    Serial.print(r_n, 3);
    Serial.print(" G=");
    Serial.print(g_n, 3);
    Serial.print(" B=");
    Serial.print(b_n, 3);
    Serial.print(" | Scores (f32): [");
    for (int i = 0; i < num_classes; ++i) {
      Serial.print(output->data.f[i], 4);
      if (i < num_classes - 1) Serial.print(", ");
    }
    Serial.print("] -> Pred: ");
    Serial.print(kClassNames[argmax]);
    Serial.print(" ");
    Serial.println(kClassEmojis[argmax]);

  } else {
    Serial.print("ERROR: Unsupported output tensor type: ");
    Serial.println(output->type);
  }

  // Small delay to avoid flooding the serial output
  delay(50);
}