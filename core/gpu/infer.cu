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
  load_weights(&nn, "/home/quan/workspace/cuda-gcc/core/gpu/weights/last.pth");

  // Assuming you have an image data loaded into `image_data`
  double *image_data; // This should be your input image data
  double *output = (double *)malloc(nn.output_size * sizeof(double));

  // Allocate memory for image_data and preprocess your image to fit the model
  image_data = (double *)malloc(FLATTENED_SIZE * sizeof(double));

  // Given have a cat.txt files which container 64 x 64 x 3 image data that has
  // been flattened, image_data would be loaded from the file
  read_csv("image/flatten/dog.txt", image_data, 1, FLATTENED_SIZE);

  // Normalize the data
  for (int i = 0; i < 10 * FLATTENED_SIZE; i++) {
    image_data[i] /= 255.0;
  }

  // Perform inference
  double *gpu_input, *gpu_z1, *gpu_a1, *gpu_z2, *gpu_a2, *gpu_w1, *gpu_b1,
      *gpu_w2, *gpu_b2;
  // You need to allocate memory for these GPU variables and copy data from CPU
  // to GPU as needed
  // cudaMalloc(&gpu_z1, nn.hidden_size * sizeof(double));
  // cudaMalloc(&gpu_a1, nn.hidden_size * sizeof(double));
  // cudaMalloc(&gpu_z2, nn.output_size * sizeof(double));
  // cudaMalloc(&gpu_a2, nn.output_size * sizeof(double));
  // cudaMalloc(&gpu_w1, nn.input_size * nn.hidden_size * sizeof(double));
  // cudaMalloc(&gpu_b1, nn.hidden_size * sizeof(double));
  // cudaMalloc(&gpu_w2, nn.hidden_size * nn.output_size * sizeof(double));
  // cudaMalloc(&gpu_b2, nn.output_size * sizeof(double));

  // cudaMemcpy(gpu_w1, nn.w1, nn.input_size * nn.hidden_size * sizeof(double),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(gpu_b1, nn.b1, nn.hidden_size * sizeof(double),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(gpu_w2, nn.w2, nn.hidden_size * nn.output_size * sizeof(double),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(gpu_b2, nn.b2, nn.output_size * sizeof(double),
  //            cudaMemcpyHostToDevice);

  predict(&nn, image_data, output, gpu_input, gpu_z1, gpu_a1, gpu_z2, gpu_a2,
          gpu_w1, gpu_b1, gpu_w2, gpu_b2);

  // Output[0] contains the prediction result
  printf("Prediction result: %f\n", output[0]);

  // Don't forget to free allocated memory and handle GPU memory as well
  free_nn(&nn);
  // Free other allocated resources
  return 0;
}