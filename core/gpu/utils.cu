/**
 * @file utils.cu
 * @brief This file contains utility functions for CUDA operations.
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-10

/**
 * @brief Computes the sigmoid function of a given value.
 * @param x The input value.
 * @return The sigmoid value of x.
 */
__device__ double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

/**
 * @brief Computes the derivative of the sigmoid function at a given value.
 * @param x The input value.
 * @return The derivative of the sigmoid function at x.
 */
__device__ double sigmoid_derivative(double x) {
  double sx = sigmoid(x);
  return sx * (1 - sx);
}

/**
 * @brief Computes the ReLU function of a given value.
 * @param x The input value.
 * @return The ReLU value of x.
 */
__device__ double relu(double x) { return x > 0 ? x : 0; }

/**
 * @brief Computes the derivative of the ReLU function at a given value.
 * @param x The input value.
 * @return The derivative of the ReLU function at x.
 */
__device__ double relu_derivative(double x) { return x > 0 ? 1 : 0; }

/**
 * @brief Computes the stable sigmoid function of a given value.
 * @param x The input value.
 * @return The stable sigmoid value of x.
 */
__device__ double stable_sigmoid(double x) {
  double sig = sigmoid(x);
  return fmax(fmin(sig, 1.0 - EPSILON), EPSILON);
}

/**
 * @brief Computes the loss between the true and predicted values.
 * @param y_true The true value.
 * @param y_pred The predicted value.
 * @return The loss between y_true and y_pred.
 */
__device__ double compute_loss(double y_true, double y_pred) {
  y_pred = fmin(fmax(y_pred, EPSILON), 1.0 - EPSILON);
  double term1 = y_true * log(y_pred);
  double term2 = (1 - y_true) * log(1 - y_pred);
  return -(term1 + term2);
}

/**
 * @brief Computes the derivative of the loss function between the true and
 * predicted values.
 * @param y_true The true value.
 * @param y_pred The predicted value.
 * @return The derivative of the loss function between y_true and y_pred.
 */
__device__ double compute_derivative_loss(double y_true, double y_pred) {
  double term1 = y_true / y_pred;
  double term2 = -(1 - y_true) / (1 - y_pred);
  return -(term1 + term2);
}

/**
 * @brief Computes the loss between the true and predicted values on the host.
 * @param y_true The true value.
 * @param y_pred The predicted value.
 * @return The loss between y_true and y_pred.
 */
double host_compute_loss(double y_true, double y_pred) {
  y_pred = fmin(fmax(y_pred, EPSILON), 1.0 - EPSILON);
  double term1 = y_true * log(y_pred);
  double term2 = (1 - y_true) * log(1 - y_pred);
  return -(term1 + term2);
}

/**
 * @brief Generates a random number from a standard normal distribution.
 * @return A random number from a standard normal distribution.
 */
double random_normal() {
  double u1 = (double)rand() / RAND_MAX;
  double u2 = (double)rand() / RAND_MAX;
  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * @brief Initializes the weights using the He initialization method.
 * @param weights The weight array.
 * @param fan_in The number of input units.
 * @param fan_out The number of output units.
 */
void he_init(double *weights, int fan_in, int fan_out) {
  double stddev = sqrt(2.0 / fan_in);
  for (int i = 0; i < fan_in * fan_out; i++) {
    weights[i] = random_normal() * stddev;
  }
}

// Utility function to read CSV files
void read_csv(const char *filename, double *data, int rows, int cols) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Error opening file");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (fscanf(file, "%lf,", &data[i * cols + j]) != 1) {
        perror("Error reading file");
        fclose(file);
        exit(EXIT_FAILURE);
      }
    }
  }

  fclose(file);
}