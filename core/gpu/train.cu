#include "neural_net.cu"

int main() {
  srand(12);

  double *train_data =
      (double *)malloc(NUM_TRAIN_IMAGES * FLATTENED_SIZE * sizeof(double));
  double *train_labels = (double *)malloc(NUM_TRAIN_IMAGES * sizeof(double));
  double *test_data =
      (double *)malloc(NUM_TEST_IMAGES * FLATTENED_SIZE * sizeof(double));
  double *test_labels = (double *)malloc(NUM_TEST_IMAGES * sizeof(double));

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
  init_nn(&nn, input_size, hidden_size, output_size);

  train(&nn, train_data, train_labels, 100, 0.01, BATCH_SIZE, NUM_TRAIN_IMAGES,
        test_data, test_labels, NUM_TEST_IMAGES);

  free_nn(&nn);
  free(train_data);
  free(train_labels);
  free(test_data);
  free(test_labels);

  return 0;
}