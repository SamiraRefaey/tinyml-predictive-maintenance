// TinyML Predictive Maintenance Model Implementation
// Generated automatically - DO NOT EDIT

#include "predictive_maintenance_model.h"
#include <math.h>

// Quantize a float value to int8
int8_t quantize_feature(float value, float min_val, float max_val) {
    if (max_val <= min_val) return 0;

    float scaled = (value - min_val) / (max_val - min_val) * 255.0f;
    int quantized = (int)(scaled + 0.5f) - 128;

    if (quantized < -128) return -128;
    if (quantized > 127) return 127;
    return (int8_t)quantized;
}

// Dequantize an int8 value to float
float dequantize_feature(int8_t quantized, float min_val, float max_val) {
    if (max_val <= min_val) return min_val;

    float scaled = (float)(quantized + 128);
    return min_val + (scaled / 255.0f) * (max_val - min_val);
}

// Predict anomaly score from float features
float predict_anomaly(const float* features) {
    float score = 0.0f;

    for (int i = 0; i < NUM_FEATURES; i++) {
        float diff = fabsf(features[i] - center[i]);
        if (scale[i] > 0.0f) {
            score += diff / scale[i];
        }
    }

    return score / NUM_FEATURES;
}

// Predict anomaly score from quantized features
float predict_anomaly_quantized(const int8_t* quantized_features) {
    float score = 0.0f;

    for (int i = 0; i < NUM_FEATURES; i++) {
        float dequantized = dequantize_feature(quantized_features[i], quant_min[i], quant_max[i]);
        float diff = fabsf(dequantized - center[i]);
        if (scale[i] > 0.0f) {
            score += diff / scale[i];
        }
    }

    return score / NUM_FEATURES;
}
