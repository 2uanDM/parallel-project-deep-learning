#include "utils.cu"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_TRAIN_IMAGES 200
#define NUM_TEST_IMAGES 50
#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define IMG_DEPTH 3
#define FLATTENED_SIZE (IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)
#define NUM_CLASSES 1
#define BATCH_SIZE 100

typedef struct {
  int input_size;
  int hidden_size;
  int output_size;

  double *w1; // weight matrix from input to hidden layer
  double *b1; // bias vector from input to hidden layer
  double *w2; // weight matrix from hidden to output layer
  double *b2; // bias vector from hidden to output layer

  double *z1; // weighted input to hidden layer
  double *a1; // activation of hidden layer
  double *z2; // weighted input to output layer
  double *a2; // activation of output layer
} NeuralNet;

void init_nn(NeuralNet *nn, int input_size, int hidden_size, int output_size) {
  nn->input_size = input_size;
  nn->hidden_size = hidden_size;
  nn->output_size = output_size;

  nn->w1 = (double *)malloc(input_size * hidden_size * sizeof(double));
  nn->b1 = (double *)malloc(hidden_size * sizeof(double));
  nn->w2 = (double *)malloc(hidden_size * output_size * sizeof(double));
  nn->b2 = (double *)malloc(output_size * sizeof(double));

  he_init(nn->w1, input_size, hidden_size);
  he_init(nn->w2, hidden_size, output_size);
}

void free_nn(NeuralNet *nn) {
  free(nn->w1);
  free(nn->b1);
  free(nn->w2);
  free(nn->b2);
}

// Forward propagation
__global__ void forward_layer1(double *gpu_input, double *gpu_z1,
                               double *gpu_a1, double *gpu_w1, double *gpu_b1,
                               int batch_size, int input_size,
                               int hidden_size) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < batch_size && Col < hidden_size) {
    double sum1 = 0.0;
    for (int k = 0; k < input_size; k++) {
      sum1 += gpu_input[Row * input_size + k] * gpu_w1[k * hidden_size + Col];
    }
    gpu_z1[Row * hidden_size + Col] = sum1 + gpu_b1[Col];

    gpu_a1[Row * hidden_size + Col] = relu(gpu_z1[Row * hidden_size + Col]);
  }
}

__global__ void forward_layer2(double *gpu_a1, double *gpu_z2, double *gpu_a2,
                               double *gpu_w2, double *gpu_b2, int batch_size,
                               int hidden_size, int output_size) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < batch_size && Col < output_size) {
    double sum2 = 0.0;
    for (int k = 0; k < hidden_size; k++) {
      sum2 += gpu_a1[Row * hidden_size + k] * gpu_w2[k * output_size + Col];
    }
    gpu_z2[Row * output_size + Col] = sum2 + gpu_b2[Col];
    gpu_a2[Row * output_size + Col] =
        stable_sigmoid(gpu_z2[Row * output_size + Col]);
  }
}

// Backward propagation
__global__ void backward1(double *gpu_loss, double *gpu_dLa2, double *gpu_da2z2,
                          double *gpu_db2, double *gpu_y_true, double *gpu_z2,
                          double *gpu_a2, int batch_size, int output_size) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < batch_size && Col < output_size) {
    gpu_loss[Row * output_size + Col] = compute_loss(
        gpu_y_true[Row * output_size + Col], gpu_a2[Row * output_size + Col]);
    gpu_dLa2[Row * output_size + Col] = compute_derivative_loss(
        gpu_y_true[Row * output_size + Col], gpu_a2[Row * output_size + Col]);
    gpu_da2z2[Row * output_size + Col] =
        sigmoid_derivative(gpu_z2[Row * output_size + Col]);
    gpu_db2[Row * output_size + Col] =
        gpu_dLa2[Row * output_size + Col] * gpu_da2z2[Row * output_size + Col];
  }
}

__global__ void backward2(double *gpu_dw2, double *gpu_dLa1, double *gpu_da1z1,
                          double *gpu_db1, double *gpu_dw1, double *gpu_input,
                          double *gpu_z1, double *gpu_a1, double *gpu_w2,
                          double *gpu_db2, int batch_size, int input_size,
                          int hidden_size) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < batch_size && Col < hidden_size) {
    gpu_dw2[Row * hidden_size + Col] =
        gpu_db2[Row] * gpu_a1[Row * hidden_size + Col];

    gpu_dLa1[Row * hidden_size + Col] = gpu_db2[Row] * gpu_w2[Col];
    gpu_da1z1[Row * hidden_size + Col] =
        relu_derivative(gpu_z1[Row * hidden_size + Col]);
    gpu_db1[Row * hidden_size + Col] =
        gpu_dLa1[Row * hidden_size + Col] * gpu_da1z1[Row * hidden_size + Col];

    for (int i = 0; i < input_size; i++) {
      gpu_dw1[Row * input_size * hidden_size + i * hidden_size + Col] =
          gpu_input[Row * input_size + i] * gpu_db1[Row * hidden_size + Col];
    }
  }
}

// Compute loss and gradient for updating
__global__ void reduction(double *gpu_dw2, double *gpu_db1, double *gpu_dw1,
                          double *gpu_w2, double *gpu_b1, double *gpu_w1,
                          double *gpu_loss, double *gpu_db2,
                          double *gpu_epoch_loss, double *gpu_b2,
                          int batch_size, int input_size, int hidden_size,
                          double learning_rate) {

  int tid = threadIdx.x;
  int node = blockIdx.x;

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (tid + s >= 100) {
        if (node == 0) {
          gpu_loss[tid] += 0;
          gpu_db2[tid] += 0;
        }

        gpu_dw2[tid * hidden_size + node] += 0;
        gpu_db1[tid * hidden_size + node] += 0;
        for (int i = 0; i < input_size; i++) {
          gpu_dw1[tid * input_size * hidden_size + i * hidden_size + node] += 0;
        }
      } else {
        if (node == 0) {
          gpu_loss[tid] += gpu_loss[tid + s];
          gpu_db2[tid] += gpu_db2[tid + s];
        }
        gpu_dw2[tid * hidden_size + node] +=
            gpu_dw2[(tid + s) * hidden_size + node];
        gpu_db1[tid * hidden_size + node] +=
            gpu_db1[(tid + s) * hidden_size + node];
        for (int i = 0; i < input_size; i++) {
          gpu_dw1[tid * input_size * hidden_size + i * hidden_size + node] +=
              gpu_dw1[(tid + s) * input_size * hidden_size + i * hidden_size +
                      node];
        }
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (node == 0) {
      gpu_db2[0] /= batch_size;
      *gpu_epoch_loss += gpu_loss[0];
      gpu_b2[0] -= learning_rate * gpu_db2[0];
    }
    gpu_dw2[node] /= batch_size;
    gpu_db1[node] /= batch_size;
    gpu_w2[node] -= learning_rate * gpu_dw2[node];
    gpu_b1[node] -= learning_rate * gpu_db1[node];
    for (int i = 0; i < input_size; i++) {
      gpu_dw1[i * hidden_size + node] /= batch_size;
      gpu_w1[i * hidden_size + node] -=
          learning_rate * gpu_dw1[i * hidden_size + node];
    }
  }
}

// Predict function
void predict(NeuralNet *nn, double *input, double *output, double *gpu_input,
             double *gpu_z1, double *gpu_a1, double *gpu_z2, double *gpu_a2,
             double *gpu_w1, double *gpu_b1, double *gpu_w2, double *gpu_b2) {
  dim3 gridSize(16, 16);
  dim3 blockSize(16, 16);
  forward_layer1<<<gridSize, blockSize>>>(gpu_input, gpu_z1, gpu_a1, gpu_w1,
                                          gpu_b1, BATCH_SIZE, nn->input_size,
                                          nn->hidden_size);
  cudaDeviceSynchronize();
  forward_layer2<<<gridSize, blockSize>>>(gpu_a1, gpu_z2, gpu_a2, gpu_w2,
                                          gpu_b2, BATCH_SIZE, nn->hidden_size,
                                          nn->output_size);
  cudaDeviceSynchronize();
  cudaMemcpy(output, gpu_a2, nn->output_size * sizeof(double),
             cudaMemcpyDeviceToHost);
}

double infer_with_probs(NeuralNet *nn, double *input, double *gpu_input,
                        double *gpu_z1, double *gpu_a1, double *gpu_z2,
                        double *gpu_a2, double *gpu_w1, double *gpu_b1,
                        double *gpu_w2, double *gpu_b2) {

  double *output = (double *)malloc(nn->output_size * sizeof(double));

  // Copy input to GPU
  cudaMemcpy(gpu_input, input, nn->input_size * sizeof(double),
             cudaMemcpyHostToDevice);

  predict(nn, input, output, gpu_input, gpu_z1, gpu_a1, gpu_z2, gpu_a2, gpu_w1,
          gpu_b1, gpu_w2, gpu_b2);

  return output[0];
}

// Test accuracy
double test_accuracy(NeuralNet *nn, double *x_test, double *y_test,
                     int data_size, double *gpu_input, double *gpu_z1,
                     double *gpu_a1, double *gpu_z2, double *gpu_a2,
                     double *gpu_w1, double *gpu_b1, double *gpu_w2,
                     double *gpu_b2) {
  int correct = 0;
  double *output = (double *)malloc(nn->output_size * sizeof(double));

  for (int i = 0; i < data_size; i++) {
    double *input = &x_test[i * nn->input_size];
    double y_true = y_test[i];

    cudaMemcpy(gpu_input, input, nn->input_size * sizeof(double),
               cudaMemcpyHostToDevice);
    predict(nn, input, output, gpu_input, gpu_z1, gpu_a1, gpu_z2, gpu_a2,
            gpu_w1, gpu_b1, gpu_w2, gpu_b2);

    if ((output[0] > 0.5) == y_true) {
      correct++;
    }
  }

  free(output);
  return (double)correct / data_size;
}

// Save weights (last and best) with error checking
void save_weights(NeuralNet *nn, const char *filename) {
  // Check if neural network pointers are initialized
  if (!nn || !nn->w1 || !nn->b1 || !nn->w2 || !nn->b2) {
    fprintf(stderr, "Error: Uninitialized neural network weights or biases.\n");
    return;
  }

  FILE *file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Error: Unable to open file %s for writing.\n", filename);
    return;
  }

  if (fwrite(nn->w1, sizeof(double), nn->input_size * nn->hidden_size, file) <
          nn->input_size * nn->hidden_size ||
      fwrite(nn->b1, sizeof(double), nn->hidden_size, file) < nn->hidden_size ||
      fwrite(nn->w2, sizeof(double), nn->hidden_size * nn->output_size, file) <
          nn->hidden_size * nn->output_size ||
      fwrite(nn->b2, sizeof(double), nn->output_size, file) < nn->output_size) {
    fprintf(stderr, "Error: Failed to write all data to file %s.\n", filename);
    fclose(file); // Ensure the file is closed even on error
    return;
  }

  fclose(file);
}

// Train the network
void train(NeuralNet *nn, double *x_train, double *y_train, int epochs,
           double learning_rate, int batch_size, int train_data_size,
           double *x_test, double *y_test, int test_data_size) {
  double *gpu_input, *gpu_z1, *gpu_a1, *gpu_z2, *gpu_a2, *gpu_w1, *gpu_b1,
      *gpu_w2, *gpu_b2, *gpu_dw1, *gpu_db1, *gpu_dw2, *gpu_db2;
  double *gpu_epoch_loss, *gpu_loss, *gpu_dLa2, *gpu_da2z2, *gpu_dLa1,
      *gpu_da1z1, *gpu_y_true;

  cudaMalloc(&gpu_input, BATCH_SIZE * nn->input_size * sizeof(double));
  cudaMalloc(&gpu_z1, BATCH_SIZE * nn->hidden_size * sizeof(double));
  cudaMalloc(&gpu_a1, BATCH_SIZE * nn->hidden_size * sizeof(double));
  cudaMalloc(&gpu_z2, BATCH_SIZE * nn->output_size * sizeof(double));
  cudaMalloc(&gpu_a2, BATCH_SIZE * nn->output_size * sizeof(double));
  cudaMalloc(&gpu_w1, nn->input_size * nn->hidden_size * sizeof(double));
  cudaMalloc(&gpu_b1, nn->hidden_size * sizeof(double));
  cudaMalloc(&gpu_w2, nn->hidden_size * nn->output_size * sizeof(double));
  cudaMalloc(&gpu_b2, nn->output_size * sizeof(double));
  cudaMalloc(&gpu_dw1,
             BATCH_SIZE * nn->input_size * nn->hidden_size * sizeof(double));
  cudaMalloc(&gpu_db1, BATCH_SIZE * nn->hidden_size * sizeof(double));
  cudaMalloc(&gpu_dw2,
             BATCH_SIZE * nn->hidden_size * nn->output_size * sizeof(double));
  cudaMalloc(&gpu_db2, BATCH_SIZE * nn->output_size * sizeof(double));

  cudaMemcpy(gpu_w1, nn->w1, nn->input_size * nn->hidden_size * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_b1, nn->b1, nn->hidden_size * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_w2, nn->w2, nn->hidden_size * nn->output_size * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_b2, nn->b2, nn->output_size * sizeof(double),
             cudaMemcpyHostToDevice);

  cudaMalloc(&gpu_loss, BATCH_SIZE * nn->output_size * sizeof(double));
  cudaMalloc(&gpu_dLa2, BATCH_SIZE * nn->output_size * sizeof(double));
  cudaMalloc(&gpu_da2z2, BATCH_SIZE * nn->output_size * sizeof(double));
  cudaMalloc(&gpu_dLa1, BATCH_SIZE * nn->hidden_size * sizeof(double));
  cudaMalloc(&gpu_da1z1, BATCH_SIZE * nn->hidden_size * sizeof(double));
  cudaMalloc(&gpu_y_true, BATCH_SIZE * sizeof(double));
  cudaMalloc(&gpu_epoch_loss, sizeof(double));

  double epoch_loss;

  // int no = (NUM_TRAIN_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;

  dim3 gridSize(16, 16);
  dim3 blockSize(16, 16);
  double time_elapsed = 0.0;

  for (int epoch = 0; epoch < epochs; epoch++) {
    double start_time = clock();
    epoch_loss = 0.0;
    cudaMemcpy(gpu_epoch_loss, &epoch_loss, sizeof(double),
               cudaMemcpyHostToDevice);

    int num_batches =
        (train_data_size + batch_size - 1) / batch_size; // Ceiling division

    for (int batch = 0; batch < num_batches; batch++) {
      int start = batch * batch_size;
      int end = start + batch_size < train_data_size ? start + batch_size
                                                     : train_data_size;
      int current_batch_size = end - start;

      double *input = x_train + start * nn->input_size;
      double *y_true = y_train + start;

      cudaMemcpy(gpu_input, input,
                 current_batch_size * nn->input_size * sizeof(double),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_y_true, y_true, current_batch_size * sizeof(double),
                 cudaMemcpyHostToDevice);

      forward_layer1<<<gridSize, blockSize>>>(gpu_input, gpu_z1, gpu_a1, gpu_w1,
                                              gpu_b1, current_batch_size,
                                              nn->input_size, nn->hidden_size);
      cudaDeviceSynchronize();

      forward_layer2<<<gridSize, blockSize>>>(gpu_a1, gpu_z2, gpu_a2, gpu_w2,
                                              gpu_b2, current_batch_size,
                                              nn->hidden_size, nn->output_size);
      cudaDeviceSynchronize();

      backward1<<<gridSize, blockSize>>>(gpu_loss, gpu_dLa2, gpu_da2z2, gpu_db2,
                                         gpu_y_true, gpu_z2, gpu_a2,
                                         current_batch_size, nn->output_size);
      cudaDeviceSynchronize();

      backward2<<<gridSize, blockSize>>>(gpu_dw2, gpu_dLa1, gpu_da1z1, gpu_db1,
                                         gpu_dw1, gpu_input, gpu_z1, gpu_a1,
                                         gpu_w2, gpu_db2, current_batch_size,
                                         nn->input_size, nn->hidden_size);
      cudaDeviceSynchronize();

      reduction<<<nn->hidden_size, 128>>>(
          gpu_dw2, gpu_db1, gpu_dw1, gpu_w2, gpu_b1, gpu_w1, gpu_loss, gpu_db2,
          gpu_epoch_loss, gpu_b2, current_batch_size, nn->input_size,
          nn->hidden_size, learning_rate);
    }
    cudaMemcpy(&epoch_loss, gpu_epoch_loss, sizeof(double),
               cudaMemcpyDeviceToHost);

    epoch_loss /= NUM_TRAIN_IMAGES;
    double end_time = clock();
    time_elapsed += (end_time - start_time) / CLOCKS_PER_SEC;
    printf("Epoch %d, Loss: %.8f Accuracy: %.2f%% Time elapse: %.4f\n",
           epoch + 1, epoch_loss,
           test_accuracy(nn, x_test, y_test, NUM_TEST_IMAGES, gpu_input, gpu_z1,
                         gpu_a1, gpu_z2, gpu_a2, gpu_w1, gpu_b1, gpu_w2,
                         gpu_b2) *
               100.0,
           time_elapsed);
  }

  // // Save weights every epoch
  // char filename[100];
  // sprintf(filename, "core/gpu/weights/last.pth");
  // save_weights(nn, filename);

  // Try to infer
  double *image_data;
  image_data = (double *)malloc(FLATTENED_SIZE * sizeof(double));
  read_csv("image/flatten/dog.txt", image_data, 1, FLATTENED_SIZE);

  for (int i = 0; i < FLATTENED_SIZE; i++) {
    image_data[i] /= 255.0;
  }

  for (int i = 0; i < 5; i++) {
    double result =
        infer_with_probs(nn, image_data, gpu_input, gpu_z1, gpu_a1, gpu_z2,
                         gpu_a2, gpu_w1, gpu_b1, gpu_w2, gpu_b2);
    printf("Predicted of dog.jpg images at %d time: %f\n", i, result);
  }

  cudaFree(gpu_input);
  cudaFree(gpu_z1);
  cudaFree(gpu_a1);
  cudaFree(gpu_z2);
  cudaFree(gpu_a2);
  cudaFree(gpu_w1);
  cudaFree(gpu_b1);
  cudaFree(gpu_w2);
  cudaFree(gpu_b2);
  cudaFree(gpu_dw1);
  cudaFree(gpu_db1);
  cudaFree(gpu_dw2);
  cudaFree(gpu_db2);
  cudaFree(gpu_loss);
  cudaFree(gpu_dLa2);
  cudaFree(gpu_da2z2);
  cudaFree(gpu_dLa1);
  cudaFree(gpu_da1z1);
  cudaFree(gpu_y_true);
  cudaFree(gpu_epoch_loss);
}