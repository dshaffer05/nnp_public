/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */
#include <cuda.h>
#include <math.h>
#include "config.h"
#include "kernels.h"



 __global__
void forward_layer(
    float *input,
    float *weights,
    float *bias,
    float *output,
    int input_size,
    int output_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < output_size) {

        float sum = bias[j];

        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + j];
        }

        output[j] = (sum > 0) ? sum : 0;
    }
}

__global__
void output_layer(
    float *input,
    float *weights,
    float *bias,
    float *output)
{
    int k = threadIdx.x;

    if (k < CLASSES) {

        float sum = bias[k];

        for (int j = 0; j < H2; j++) {
            sum += input[j] * weights[j * CLASSES + k];
        }

        output[k] = sum;
    }
}

__global__
void softmax_kernel(
    float *z,
    float *out,
    int len)
{
    float max = z[0];

    for (int i = 1; i < len; i++) {
        if (z[i] > max) max = z[i];
    }

    float sum = 0.0f;

    for (int i = 0; i < len; i++) {
        out[i] = expf(z[i] - max);
        sum += out[i];
    }

    for (int i = 0; i < len; i++) {
        out[i] /= sum;
    }
}

__global__
void delta3_kernel(
    float *label,
    float *outa,
    float *delta3)
{
    int k = threadIdx.x;

    if (k < CLASSES) {
        delta3[k] = label[k] - outa[k];
    }
}

__global__
void delta2_kernel(
    float *delta3,
    float *W3,
    float *h2a,
    float *delta2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < H2) {

        float err = 0.0f;

        for (int k = 0; k < CLASSES; k++) {
            err += delta3[k] * W3[j * CLASSES + k];
        }

        delta2[j] = err * (h2a[j] > 0 ? 1.0f : 0.0f);
    }
}

__global__
void delta1_kernel(
    float *delta2,
    float *W2,
    float *h1a,
    float *delta1)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < H1) {

        float err = 0.0f;

        for (int k = 0; k < H2; k++) {
            err += delta2[k] * W2[j * H2 + k];
        }

        delta1[j] = err * (h1a[j] > 0 ? 1.0f : 0.0f);
    }
}

__global__
void update_W3(
    float *W3,
    float *delta3,
    float *h2a)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = H2 * CLASSES;

    if (idx < total) {

        int j = idx / CLASSES;
        int k = idx % CLASSES;

        W3[idx] += LR * delta3[k] * h2a[j];
    }
}

__global__
void update_W2(
    float *W2,
    float *delta2,
    float *h1a)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = H1 * H2;

    if (idx < total) {

        int j = idx / H2;
        int k = idx % H2;

        W2[idx] += LR * delta2[k] * h1a[j];
    }
}

__global__
void update_W1(
    float *W1,
    float *delta1,
    float *input)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = SIZE * H1;

    if (idx < total) {

        int i = idx / H1;
        int j = idx % H1;

        W1[idx] += LR * delta1[j] * input[i];
    }
}

__global__
void update_bias(
    float *bias,
    float *delta,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        bias[idx] += LR * delta[idx];
    }
}