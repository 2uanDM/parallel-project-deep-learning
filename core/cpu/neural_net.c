#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Import from utils.c
#include "utils.c"

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

// Initialize the neural

void init_neural_net(NeuralNet *nn, int input_size, int hidden_size,
                     int output_size) {
  nn->input_size = input_size;
  nn->hidden_size = hidden_size;
  nn->output_size = output_size;

  nn->w1 = (double *)malloc(input_size * hidden_size * sizeof(double));
  nn->b1 = (double *)malloc(hidden_size * sizeof(double));
  nn->w2 = (double *)malloc(hidden_size * output_size * sizeof(double));
  nn->b2 = (double *)malloc(output_size * sizeof(double));

  nn->z1 = (double *)malloc(hidden_size * sizeof(double));
  nn->a1 = (double *)malloc(hidden_size * sizeof(double));
  nn->z2 = (double *)malloc(output_size * sizeof(double));
  nn->a2 = (double *)malloc(output_size * sizeof(double));

  if (!nn->w1 || !nn->b1 || !nn->w2 || !nn->b2 || !nn->z1 || !nn->a1 ||
      !nn->z2 || !nn->a2) {
    perror("Error allocating memory");
    exit(EXIT_FAILURE);
  }

  he_init(nn->w1, input_size, hidden_size);
  he_init(nn->w2, hidden_size, output_size);
}

void forward(NeuralNet *nn, double *input) {
  for (int i = 0; i < nn->hidden_size; i++) {
    nn->z1[i] = 0.0;
    for (int j = 0; j < nn->input_size; j++) {
      nn->z1[i] += input[j] * nn->w1[j * nn->hidden_size + i];
    }
    nn->z1[i] += nn->b1[i];
    nn->a1[i] = relu(nn->z1[i]);
  }

  for (int i = 0; i < nn->output_size; i++) {
    nn->z2[i] = 0.0;
    for (int j = 0; j < nn->hidden_size; j++) {
      nn->z2[i] += nn->a1[j] * nn->w2[j * nn->output_size + i];
    }
    nn->z2[i] += nn->b2[i];
    nn->a2[i] = stable_sigmoid(nn->z2[i]);
  }
}

double compute_loss(double *y_true, double *y_pred, int size) {
  double loss = 0.0;
  for (int i = 0; i < size; i++) {
    double pred = y_pred[i];
    // Avoid log(0) and log(1)
    if (pred < 1e-8)
      pred = 1e-8;
    if (pred > 1 - 1e-8)
      pred = 1 - 1e-8;

    double true_value = y_true[i];
    double term1 = true_value * log(pred);
    double term2 = (1 - true_value) * log(1 - pred);

    loss += -(term1 + term2);
  }
  return loss / size;
}

void backward(NeuralNet *nn, double *input, double *y_true, double *dw1,
              double *db1, double *dw2, double *db2) {
  double dL_da2[nn->output_size];
  double da2_dz2[nn->output_size];
  double dL_da1[nn->hidden_size];
  double da1_dz1[nn->hidden_size];

  for (int i = 0; i < nn->output_size; i++) {
    dL_da2[i] =
        -(*(y_true + i) / nn->a2[i] - (1 - *(y_true + i)) / (1 - nn->a2[i]));
    da2_dz2[i] = sigmoid_derivative(nn->z2[i]);
    db2[i] = dL_da2[i] * da2_dz2[i];
    for (int j = 0; j < nn->hidden_size; j++) {
      dw2[j * nn->output_size + i] = nn->a1[j] * db2[i];
    }
  }

  for (int i = 0; i < nn->hidden_size; i++) {
    dL_da1[i] = 0.0;
    for (int j = 0; j < nn->output_size; j++) {
      dL_da1[i] += dL_da2[j] * da2_dz2[j] * nn->w2[i * nn->output_size + j];
    }
    da1_dz1[i] = relu_derivative(nn->z1[i]);
    db1[i] = dL_da1[i] * da1_dz1[i];
    for (int j = 0; j < nn->input_size; j++) {
      dw1[j * nn->hidden_size + i] = input[j] * db1[i];
    }
  }
}

void update_params(NeuralNet *nn, double *dw1, double *db1, double *dw2,
                   double *db2, double learning_rate, int batch_size) {
  for (int i = 0; i < nn->input_size * nn->hidden_size; i++) {
    nn->w1[i] -= (learning_rate * dw1[i]) / batch_size;
  }
  for (int i = 0; i < nn->hidden_size; i++) {
    nn->b1[i] -= (learning_rate * db1[i]) / batch_size;
  }
  for (int i = 0; i < nn->hidden_size * nn->output_size; i++) {
    nn->w2[i] -= (learning_rate * dw2[i]) / batch_size;
  }
  for (int i = 0; i < nn->output_size; i++) {
    nn->b2[i] -= (learning_rate * db2[i]) / batch_size;
  }
}

void predict(NeuralNet *nn, double *input, double *output) {
  forward(nn, input);
  for (int i = 0; i < nn->output_size; i++) {
    output[i] = nn->a2[i] > 0.5 ? 1.0 : 0.0;
  }
}

double test_accuracy(NeuralNet *nn, double *x_test, double *y_test,
                     int data_size) {
  int correct = 0;
  double *output = (double *)malloc(nn->output_size * sizeof(double));

  for (int i = 0; i < data_size; i++) {
    double *input = &x_test[i * nn->input_size];
    double y_true = y_test[i];

    predict(nn, input, output);

    if (output[0] == y_true) {
      correct++;
    }
  }

  free(output);
  return (double)correct / data_size;
}

void train(NeuralNet *nn, double *x_train, double *y_train, int epochs,
           double learning_rate, int batch_size, int train_data_size,
           double **best_w1, double **best_b1, double **best_w2,
           double **best_b2, double *x_test, double *y_test,
           int test_data_size) {
  double *dw1 =
      (double *)malloc(nn->input_size * nn->hidden_size * sizeof(double));
  double *db1 = (double *)malloc(nn->hidden_size * sizeof(double));
  double *dw2 =
      (double *)malloc(nn->hidden_size * nn->output_size * sizeof(double));
  double *db2 = (double *)malloc(nn->output_size * sizeof(double));
  double *batch_dw1 =
      (double *)malloc(nn->input_size * nn->hidden_size * sizeof(double));
  double *batch_db1 = (double *)malloc(nn->hidden_size * sizeof(double));
  double *batch_dw2 =
      (double *)malloc(nn->hidden_size * nn->output_size * sizeof(double));
  double *batch_db2 = (double *)malloc(nn->output_size * sizeof(double));

  if (!dw1 || !db1 || !dw2 || !db2) {
    perror("Error allocating memory for gradients");
    exit(EXIT_FAILURE);
  }

  double min_loss = DBL_MAX;
  for (int epoch = 0; epoch < epochs; epoch++) {
    double epoch_loss = 0.0;
    int num_batches =
        (train_data_size + batch_size - 1) / batch_size; // Ceiling division

    for (int batch = 0; batch < num_batches; batch++) {
      int start = batch * batch_size;
      int end = start + batch_size < train_data_size ? start + batch_size
                                                     : train_data_size;
      int current_batch_size = end - start;

      memset(dw1, 0, nn->input_size * nn->hidden_size * sizeof(double));
      memset(db1, 0, nn->hidden_size * sizeof(double));
      memset(dw2, 0, nn->hidden_size * nn->output_size * sizeof(double));
      memset(db2, 0, nn->output_size * sizeof(double));

      for (int i = start; i < end; i++) {
        double *input = x_train + i * nn->input_size;
        double *y_true = y_train + i;

        forward(nn, input);
        epoch_loss += compute_loss(y_true, nn->a2, nn->output_size);
        memset(batch_dw1, 0, nn->input_size * nn->hidden_size * sizeof(double));
        memset(batch_db1, 0, nn->hidden_size * sizeof(double));
        memset(batch_dw2, 0,
               nn->hidden_size * nn->output_size * sizeof(double));
        memset(batch_db2, 0, nn->output_size * sizeof(double));
        backward(nn, input, y_true, batch_dw1, batch_db1, batch_dw2, batch_db2);

        for (int j = 0; j < nn->input_size * nn->hidden_size; j++) {
          dw1[j] += batch_dw1[j];
        }
        for (int j = 0; j < nn->hidden_size; j++) {
          db1[j] += batch_db1[j];
        }
        for (int j = 0; j < nn->hidden_size * nn->output_size; j++) {
          dw2[j] += batch_dw2[j];
        }
        for (int j = 0; j < nn->output_size; j++) {
          db2[j] += batch_db2[j];
        }
      }
      update_params(nn, dw1, db1, dw2, db2, learning_rate, current_batch_size);
    }

    epoch_loss /= train_data_size;
    printf("Epoch %d, Loss: %.8f Accuracy: %.2f%%\n", epoch + 1, epoch_loss,
           test_accuracy(nn, x_test, y_test, test_data_size) * 100.0);

    if (epoch_loss < min_loss) {
      min_loss = epoch_loss;
      memcpy(*best_w1, nn->w1,
             nn->input_size * nn->hidden_size * sizeof(double));
      memcpy(*best_b1, nn->b1, nn->hidden_size * sizeof(double));
      memcpy(*best_w2, nn->w2,
             nn->hidden_size * nn->output_size * sizeof(double));
      memcpy(*best_b2, nn->b2, nn->output_size * sizeof(double));
    }
  }

  free(dw1);
  free(db1);
  free(dw2);
  free(db2);
}
