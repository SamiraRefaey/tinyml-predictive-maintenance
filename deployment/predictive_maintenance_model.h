// TinyML Predictive Maintenance Model
// Generated automatically - DO NOT EDIT

#ifndef PREDICTIVE_MAINTENANCE_MODEL_H
#define PREDICTIVE_MAINTENANCE_MODEL_H

#include <stdint.h>

#define NUM_FEATURES 21
#define MODEL_THRESHOLD 1.728103f

// Model parameters
static const float center[NUM_FEATURES] = {
    2.613844f,
    0.097657f,
    2.792863f,
    2.437813f,
    2.615628f,
    0.020620f,
    -0.619210f,
    41.206090f,
    0.471065f,
    41.918432f,
    40.484075f,
    41.208690f,
    -0.016624f,
    -1.217850f,
    2.046560f,
    0.023007f,
    2.083101f,
    2.009637f,
    2.046685f,
    -0.039964f,
    -1.169582f
};

static const float scale[NUM_FEATURES] = {
    0.037730f,
    0.013615f,
    0.029157f,
    0.048984f,
    0.037570f,
    0.373442f,
    0.524717f,
    0.092106f,
    0.040606f,
    0.065177f,
    0.066794f,
    0.092111f,
    0.306787f,
    0.230124f,
    0.004777f,
    0.002206f,
    0.003900f,
    0.003542f,
    0.004772f,
    0.200874f,
    0.301617f
};

// Quantization ranges
static const float quant_min[NUM_FEATURES] = {
    2.563235f,
    0.075321f,
    2.733348f,
    2.381643f,
    2.564770f,
    -0.770790f,
    -1.456180f,
    40.997947f,
    0.299443f,
    41.755991f,
    40.400650f,
    41.000156f,
    -0.674925f,
    -1.544029f,
    2.036063f,
    0.017372f,
    2.070958f,
    2.005707f,
    2.036200f,
    -0.477146f,
    -1.519127f
};

static const float quant_max[NUM_FEATURES] = {
    3.563401f,
    0.310508f,
    3.930722f,
    3.152979f,
    3.568588f,
    0.770786f,
    0.148169f,
    45.098854f,
    1.873810f,
    45.756269f,
    44.401825f,
    45.100894f,
    0.566342f,
    -0.506282f,
    2.321770f,
    0.129878f,
    2.356650f,
    2.282253f,
    2.321880f,
    0.662993f,
    -0.432718f
};

// Function declarations
int8_t quantize_feature(float value, float min_val, float max_val);
float dequantize_feature(int8_t quantized, float min_val, float max_val);
float predict_anomaly(const float* features);
float predict_anomaly_quantized(const int8_t* quantized_features);

#endif // PREDICTIVE_MAINTENANCE_MODEL_H
