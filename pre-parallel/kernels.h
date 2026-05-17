/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();


__global__
void forward_layer(
    float *input,
    float *weights,
    float *bias,
    float *output,
    int input_size,
    int output_size);

__global__
void output_layer(
    float *input,
    float *weights,
    float *bias,
    float *output);

__global__
void softmax_kernel(
    float *z,
    float *out,
    int len);

__global__
void delta3_kernel(
    float *label,
    float *outa,
    float *delta3);

__global__
void delta2_kernel(
    float *delta3,
    float *W3,
    float *h2a,
    float *delta2);

__global__
void delta1_kernel(
    float *delta2,
    float *W2,
    float *h1a,
    float *delta1);

__global__
void update_W3(
    float *W3,
    float *delta3,
    float *h2a);

__global__
void update_W2(
    float *W2,
    float *delta2,
    float *h1a);

__global__
void update_W1(
    float *W1,
    float *delta1,
    float *input);

__global__
void update_bias(
    float *bias,
    float *delta,
    int size);