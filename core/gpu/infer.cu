#include "neural_net.cu"

// Function to load weights into the neural network
void load_weights(NeuralNet *nn, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error: Unable to open file %s for reading.\n", filename);
    return;
  }
  if (fread(nn->w1, sizeof(double), nn->input_size * nn->hidden_size, file) <
          nn->input_size * nn->hidden_size ||
      fread(nn->b1, sizeof(double), nn->hidden_size, file) < nn->hidden_size ||
      fread(nn->w2, sizeof(double), nn->hidden_size * nn->output_size, file) <
          nn->hidden_size * nn->output_size ||
      fread(nn->b2, sizeof(double), nn->output_size, file) < nn->output_size) {
    fprintf(stderr, "Error: Failed to read all data from file %s.\n", filename);
  }
  fclose(file);
}

// Example usage
int main() {
  NeuralNet nn;
  init_nn(&nn, FLATTENED_SIZE, 128,
          1); // Initialize the neural network structure
  load_weights(&nn,
               "/home/quan/workspace/cuda-gcc/core/gpu/weights/epoch_99.pth");

  // Assuming you have an image data loaded into `image_data`
  double *image_data; // This should be your input image data
  double output[1];   // Assuming binary classification for simplicity

  // Allocate memory for image_data and preprocess your image to fit the model
  image_data = (double *)malloc(FLATTENED_SIZE * sizeof(double));

  //   // Print the neural network weights
  //   printf("w1: ");
  //   for (int i = 0; i < nn.input_size * nn.hidden_size; i++) {
  //     printf("%f ", nn.w1[i]);
  //   }
  //   printf("\n");

  //   printf("b1: ");
  //   for (int i = 0; i < nn.hidden_size; i++) {
  //     printf("%f ", nn.b1[i]);
  //   }
  //   printf("\n");

  //   printf("w2: ");
  //   for (int i = 0; i < nn.hidden_size * nn.output_size; i++) {
  //     printf("%f ", nn.w2[i]);
  //   }
  //   printf("\n");

  //   printf("b2: ");
  //   for (int i = 0; i < nn.output_size; i++) {
  //     printf("%f ", nn.b2[i]);
  //   }
  //   printf("\n");

  // Given have a cat.txt files which container 64 x 64 x 3 image data that has
  // been flattened, image_data would be loaded from the file
  //   read_csv("image/flatten/dog.txt", image_data, 1, FLATTENED_SIZE);
  read_csv("catfinal/test_images_float32.csv", image_data, 1, FLATTENED_SIZE);

  // Perform inference
  double *gpu_input, *gpu_z1, *gpu_a1, *gpu_z2, *gpu_a2, *gpu_w1, *gpu_b1,
      *gpu_w2, *gpu_b2;
  // You need to allocate memory for these GPU variables and copy data from CPU
  // to GPU as needed
  predict(&nn, image_data, output, gpu_input, gpu_z1, gpu_a1, gpu_z2, gpu_a2,
          gpu_w1, gpu_b1, gpu_w2, gpu_b2);

  // Output[0] contains the prediction result
  printf("Prediction result: %f\n", output[0]);

  // Don't forget to free allocated memory and handle GPU memory as well
  free_nn(&nn);
  // Free other allocated resources
  return 0;
}