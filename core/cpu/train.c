#include <float.h>
#include <libconfig.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Import from utils.c
#include "neural_net.c"

#define NUM_TRAIN_IMAGES 200
#define NUM_TEST_IMAGES 50
#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define IMG_DEPTH 3
#define FLATTENED_SIZE (IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)
#define NUM_CLASSES 1
#define BATCH_SIZE 10

int main() {
  srand(12);

  double *train_data =
      (double *)malloc(NUM_TRAIN_IMAGES * FLATTENED_SIZE * sizeof(double));
  double *train_labels = (double *)malloc(NUM_TRAIN_IMAGES * sizeof(double));
  double *test_data =
      (double *)malloc(NUM_TEST_IMAGES * FLATTENED_SIZE * sizeof(double));
  double *test_labels = (double *)malloc(NUM_TEST_IMAGES * sizeof(double));

  if (!train_data || !train_labels || !test_data || !test_labels) {
    perror("Error allocating memory for data");
    exit(EXIT_FAILURE);
  }

  read_csv("catfinal/final_train_images_float32.csv", train_data,
           NUM_TRAIN_IMAGES, FLATTENED_SIZE);
  read_csv("catfinal/final_train_labels_float32.csv", train_labels,
           NUM_TRAIN_IMAGES, 1);
  read_csv("catfinal/test_images_float32.csv", test_data, NUM_TEST_IMAGES,
           FLATTENED_SIZE);
  read_csv("catfinal/test_labels_float32.csv", test_labels, NUM_TEST_IMAGES, 1);

  // Normalize the data
  for (int i = 0; i < NUM_TRAIN_IMAGES * FLATTENED_SIZE; i++) {
    train_data[i] /= 255.0;
  }
  for (int i = 0; i < NUM_TEST_IMAGES * FLATTENED_SIZE; i++) {
    test_data[i] /= 255.0;
  }

  int input_size = FLATTENED_SIZE;
  int hidden_size = 128;
  int output_size = 1;

  NeuralNet nn;
  init_neural_net(&nn, input_size, hidden_size, output_size);

  double *best_w1 = (double *)malloc(input_size * hidden_size * sizeof(double));
  double *best_b1 = (double *)malloc(hidden_size * sizeof(double));
  double *best_w2 =
      (double *)malloc(hidden_size * output_size * sizeof(double));
  double *best_b2 = (double *)malloc(output_size * sizeof(double));

  if (!best_w1 || !best_b1 || !best_w2 || !best_b2) {
    perror("Error allocating memory for best parameters");
    exit(EXIT_FAILURE);
  }

  train(&nn, train_data, train_labels, 100, 0.01, BATCH_SIZE, NUM_TRAIN_IMAGES,
        &best_w1, &best_b1, &best_w2, &best_b2, test_data, test_labels,
        NUM_TEST_IMAGES);

  // Load the best parameters into the network
  memcpy(nn.w1, best_w1, input_size * hidden_size * sizeof(double));
  memcpy(nn.b1, best_b1, hidden_size * sizeof(double));
  memcpy(nn.w2, best_w2, hidden_size * output_size * sizeof(double));
  memcpy(nn.b2, best_b2, output_size * sizeof(double));

  double accuracy = test_accuracy(&nn, test_data, test_labels, NUM_TEST_IMAGES);
  printf("Test Accuracy: %.2lf%%\n", accuracy * 100);

  // Free allocated memory
  free(nn.w1);
  free(nn.b1);
  free(nn.w2);
  free(nn.b2);
  free(nn.z1);
  free(nn.a1);
  free(nn.z2);
  free(nn.a2);
  free(train_data);
  free(train_labels);
  free(test_data);
  free(test_labels);
  free(best_w1);
  free(best_b1);
  free(best_w2);
  free(best_b2);

  return 0;
}