/**
 * @file utils.c
 * @brief This file contains utility functions for mathematical operations.
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define EPSILON 1e-10

/**
 * @brief Calculates the sigmoid function of a given value.
 *
 * The sigmoid function is defined as 1 / (1 + exp(-x)).
 *
 * @param x The input value.
 * @return The sigmoid value of the input.
 */
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

/**
 * @brief Calculates the stable sigmoid function of a given value.
 *
 * The stable sigmoid function is defined as fmax(fmin(sigmoid(x), 1.0 -
 * EPSILON), EPSILON), where EPSILON is a small value to avoid numerical
 * instability.
 *
 * @param x The input value.
 * @return The stable sigmoid value of the input.
 */
double stable_sigmoid(double x) {
  double sig = sigmoid(x);
  return fmax(fmin(sig, 1.0 - EPSILON), EPSILON);
}

/**
 * @brief Calculates the derivative of the sigmoid function at a given value.
 *
 * The derivative of the sigmoid function is defined as sigmoid(x) * (1 -
 * sigmoid(x)).
 *
 * @param x The input value.
 * @return The derivative of the sigmoid function at the input.
 */
double sigmoid_derivative(double x) {
  double sx = sigmoid(x);
  return sx * (1 - sx);
}

/**
 * @brief Calculates the rectified linear unit (ReLU) of a given value.
 *
 * The ReLU function is defined as max(x, 0).
 *
 * @param x The input value.
 * @return The ReLU value of the input.
 */
double relu(double x) { return x > 0 ? x : 0.0; }

/**
 * @brief Calculates the derivative of the rectified linear unit (ReLU) at a
 * given value.
 *
 * The derivative of the ReLU function is defined as 1.0 if x > 0, and 0.0
 * otherwise.
 *
 * @param x The input value.
 * @return The derivative of the ReLU function at the input.
 */
double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

/**
 * @brief Generates a random number from a standard normal distribution.
 *
 * @return The random number.
 */
double random_normal() {
  double u1 = (double)rand() / RAND_MAX;
  double u2 = (double)rand() / RAND_MAX;
  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * @brief Initializes the weights array with values drawn from a random normal
 * distribution.
 *
 * The standard deviation of the distribution is calculated based on the fan_in
 * value.
 *
 * @param weights The array of weights to be initialized.
 * @param fan_in The number of input units.
 * @param fan_out The number of output units.
 */
void he_init(double *weights, int fan_in, int fan_out) {
  double stddev = sqrt(2.0 / fan_in);
  for (int i = 0; i < fan_in * fan_out; i++) {
    weights[i] = random_normal() * stddev;
  }
}

/**
 * @brief Reads data from a CSV file and stores it in a 2D array.
 *
 * The CSV file should have rows and columns separated by commas.
 *
 * @param filename The name of the CSV file.
 * @param data The 2D array to store the data.
 * @param rows The number of rows in the CSV file.
 * @param cols The number of columns in the CSV file.
 */
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