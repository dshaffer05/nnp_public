/*
    nnp.cu

    Created on: Nov 9, 2025
    Serial implementation of a simple feedforward neural network for MNIST digit classification.

    Network architecture:
    - Input layer: 784 neurons (28x28 pixels)
    - Hidden layer 1: 128 neurons, ReLU activation
    - Hidden layer 2: 64 neurons, ReLU activation
    - Output layer: 10 neurons, Softmax activation

    Training:
    - Loss function: Categorical Cross-Entropy
    - Optimizer: Stochastic Gradient Descent (SGD)
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include "kernels.h"


/* Activation functions for relu layers
* Arguments:
*   x: input value
* Returns:
*   activated value based on ReLU function 
*/
float relu(float x) { return x > 0 ? x : 0; }

/* Derivative of ReLU activation function
* Arguments:
*   y: output value from ReLU function
* Returns:
*   derivative value
*/
float drelu(float y) { return y > 0 ? 1 : 0; }

/* Softmax activation function
* Arguments:
*   z: input array
*   out: output array to store softmax results
*   len: length of the input/output arrays
*/ 
void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

/* Initialize weights with small random values
* Arguments:
*   w: weight array to initialize
*   size: number of weights
*/
void init_weights(float *w, int size) {
    for (int i=0;i<size;i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
}



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


/* Train the model using stochastic gradient descent 
* Arguments:
*   model (out): pointer to the MODEL structure which holds network parameters. It is populated by this function.
* Returns:
*   None
*/
void train_model(MODEL* model){
    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    float *d_W3, *d_b3;
    cudaMalloc(&d_W1, SIZE * H1 * sizeof(float));
    cudaMalloc(&d_b1, H1 * sizeof(float));
    cudaMalloc(&d_W2, H1 * H2 * sizeof(float));
    cudaMalloc(&d_b2, H2 * sizeof(float));
    cudaMalloc(&d_W3, H2 * CLASSES * sizeof(float));
    cudaMalloc(&d_b3, CLASSES * sizeof(float));


    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);

    cudaMemcpy(d_W1, model->W1,
                SIZE * H1 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, model->b1,
               H1 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, model->W2,
               H1 * H2 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, model->b2,
               H2 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, model->W3,
               H2 * CLASSES * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, model->b3,
               CLASSES * sizeof(float),
               cudaMemcpyHostToDevice);

    float *d_input;
    float *d_label;
    float *d_h1a;
    float *d_h2a;
    float *d_out;
    float *d_outa;
    float *d_delta1;
    float *d_delta2;
    float *d_delta3;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_label, CLASSES * sizeof(float));
    cudaMalloc(&d_h1a, H1 * sizeof(float));
    cudaMalloc(&d_h2a, H2 * sizeof(float));
    cudaMalloc(&d_out, CLASSES * sizeof(float));
    cudaMalloc(&d_outa, CLASSES * sizeof(float));
    cudaMalloc(&d_delta1, H1 * sizeof(float));
    cudaMalloc(&d_delta2, H2 * sizeof(float));
    cudaMalloc(&d_delta3, CLASSES * sizeof(float));

    for (int epoch=0; epoch<EPOCHS; epoch++) {
        float loss=0;
        for (int n=0; n<NUM_TRAIN; n++) {
            cudaMemcpy(d_input,
                       train_data[n],
                       SIZE * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_label,
                       train_label[n],
                       CLASSES * sizeof(float),
                       cudaMemcpyHostToDevice);
            // ---------- Forward ----------
            forward_layer<<<(H1 + 255)/256, 256>>>(
                d_input,
                d_W1,
                d_b1,
                d_h1a,
                SIZE,
                H1
            );

            forward_layer<<<(H2 + 255)/256, 256>>>(
                d_h1a,
                d_W2,
                d_b2,
                d_h2a,
                H1,
                H2
            );

            output_layer<<<1, CLASSES>>>(
                d_h2a,
                d_W3,
                d_b3,
                d_out
            );

            softmax_kernel<<<1,1>>>(
                d_out,
                d_outa,
                CLASSES
            );


            // ---------- Backprop ----------
            delta3_kernel<<<1, CLASSES>>>(
                d_label,
                d_outa,
                d_delta3
            );

            delta2_kernel<<<(H2 + 255)/256, 256>>>(
                d_delta3,
                d_W3,
                d_h2a,
                d_delta2
            );

            delta1_kernel<<<(H1 + 255)/256, 256>>>(
                d_delta2,
                d_W2,
                d_h1a,
                d_delta1
            );

            // ---------- Update ----------
            update_W3<<<(H2*CLASSES + 255)/256, 256>>>(
                d_W3,
                d_delta3,
                d_h2a
            );

            update_bias<<<1, CLASSES>>>(
                d_b3,
                d_delta3,
                CLASSES
            );

            update_W2<<<(H1*H2 + 255)/256, 256>>>(
                d_W2,
                d_delta2,
                d_h1a
            );

            update_bias<<<(H2 + 255)/256, 256>>>(
                d_b2,
                d_delta2,
                H2
            );

            update_W1<<<(SIZE*H1 + 255)/256, 256>>>(
                d_W1,
                d_delta1,
                d_input
            );

            update_bias<<<(H1 + 255)/256, 256>>>(
                d_b1,
                d_delta1,
                H1
            );

            cudaDeviceSynchronize();
        }

        printf("Epoch %d complete\n", epoch);
    }

    cudaMemcpy(model->W1, d_W1,
               SIZE * H1 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(model->b1, d_b1,
               H1 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(model->W2, d_W2,
               H1 * H2 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(model->b2, d_b2,
               H2 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(model->W3, d_W3,
               H2 * CLASSES * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(model->b3, d_b3,
               CLASSES * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ---------------- Cleanup ----------------

    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_W3);
    cudaFree(d_b3);
    cudaFree(d_input);
    cudaFree(d_label);
    cudaFree(d_h1a);
    cudaFree(d_h2a);
    cudaFree(d_out);
    cudaFree(d_outa);
    cudaFree(d_delta1);
    cudaFree(d_delta2);
    cudaFree(d_delta3);
}

/* Save the trained model to a binary file
* Arguments:
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None
*/
void save_model(MODEL* model){
	FILE *f = fopen("model.bin", "wb");
	fwrite(model->W1, sizeof(float), SIZE*H1, f);
	fwrite(model->b1, sizeof(float), H1, f);
	fwrite(model->W2, sizeof(float), H1*H2, f);
	fwrite(model->b2, sizeof(float), H2, f);
	fwrite(model->W3, sizeof(float), H2*CLASSES, f);
	fwrite(model->b3, sizeof(float), CLASSES,f);
	fclose(f);
}

/* Load the trained model from a binary file
* Arguments:
*   model (out): pointer to the MODEL structure to populate with loaded weights and biases
* Returns:
*   None
*/
void load_model(MODEL* model){
	FILE *f = fopen("model.bin", "rb");
	fread(model->W1, sizeof(float), SIZE*H1, f);
	fread(model->b1, sizeof(float), H1, f);
	fread(model->W2, sizeof(float), H1*H2, f);
	fread(model->b2, sizeof(float), H2, f);
	fread(model->W3, sizeof(float), H2*CLASSES, f);
	fread(model->b3, sizeof(float), CLASSES, f);
	fclose(f);
}

/* Predict the class of a given input image
* Arguments:
*   x: input image array (flattened 28x28 pixels)
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None (prints predicted class and confidence)
*/
void predict(float *x, MODEL* model){
    float h1[H1], h1a[H1], h2[H2], h2a[H2], out[CLASSES], outa[CLASSES];

    // forward pass
    for (int j=0;j<H1;j++){ h1[j]=model->b1[j]; for(int i=0;i<SIZE;i++) h1[j]+=x[i]*model->W1[i*H1+j]; h1a[j]=relu(h1[j]); }
    for (int j=0;j<H2;j++){ h2[j]=model->b2[j]; for(int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j]; h2a[j]=relu(h2[j]); }
    for (int k=0;k<CLASSES;k++){ out[k]=model->b3[k]; for(int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k]; }
    softmax(out,outa,CLASSES);

    // print predicted class
    int pred=0; float max=outa[0];
    for(int k=1;k<CLASSES;k++) if(outa[k]>max){ max=outa[k]; pred=k; }
    printf("Predicted digit: %d (confidence %.2f)\n", pred, max);
}


