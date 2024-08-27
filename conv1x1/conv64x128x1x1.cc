#include <cnpy.h>

#include <fstream>
#include <iostream>

#define CHANNELS 128
#define HEIGHT 224
#define WIDTH 224
#define NUM_FILTERS 64
#define KERNEL_HEIGHT 1
#define KERNEL_WIDTH 1
#define STRIDE 2
#define PADDING 1

using namespace std;

int main() {
  //  input  : 1x128x224x224 (nchw)
  //  kernel : 64x128x1x1 (oihw)
  //  output : 1x64x224x224 [considering stride 1 and padding 0]

  int padded_height = HEIGHT + 2 * PADDING;
  int padded_width = WIDTH + 2 * PADDING;

  float *input = new float[CHANNELS * padded_height * padded_width];
  float *filters =
      new float[NUM_FILTERS * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH];
  float *bias = new float[NUM_FILTERS];

  for (int i = 0; i < CHANNELS * padded_height * padded_width; i++) {
    input[i] = 0;
  }
  // pad the input if PADDING is not 0
  int num = 0;
  for (int i = 0; i < CHANNELS; i++) {
    for (int j = 0; j < HEIGHT; j++) {
      for (int k = 0; k < WIDTH; k++) {
        input[i * padded_height * padded_width + (j + PADDING) * padded_width +
              (k + PADDING)] = (++num);
      }
    }
  }

  // Fill weights with 1 and biases with 0
  for (int f = 0; f < NUM_FILTERS; f++) {
    for (int ch = 0; ch < CHANNELS; ch++) {
      for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
          filters[f * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH +
                  ch * KERNEL_HEIGHT * KERNEL_WIDTH + i * KERNEL_WIDTH + j] = 1;
        }
      }
    }
  }
  for (int i = 0; i < NUM_FILTERS; i++) {
    bias[i] = 0;
  }
  cout << "Initialization Done" << endl;

  // output size
  int output_height = (padded_height - KERNEL_HEIGHT) / STRIDE + 1;
  int output_width = (padded_width - KERNEL_WIDTH) / STRIDE + 1;
  float *output_matrix = new float[NUM_FILTERS * output_height * output_width];

  cout << "Output shape : [ " << NUM_FILTERS << " ][ " << output_height
       << " ][ " << output_width << " ]" << endl;

  // Perform convolution
  for (int f = 0; f < NUM_FILTERS; f++) {
    for (int i = 0; i < output_height; i++) {
      for (int j = 0; j < output_width; j++) {
        float output = bias[f];
        for (int ch = 0; ch < CHANNELS; ch++) {
          for (int k = 0; k < KERNEL_HEIGHT; k++) {
            for (int l = 0; l < KERNEL_WIDTH; l++) {
              int input_row = i * STRIDE + k;
              int input_col = j * STRIDE + l;
              output += input[ch * padded_height * padded_width +
                              input_row * padded_width + input_col] *
                        filters[f * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH +
                                ch * KERNEL_HEIGHT * KERNEL_WIDTH +
                                k * KERNEL_WIDTH + l];
            }
          }
        }
        output_matrix[f * output_height * output_width + i * output_width + j] =
            output;
      }
    }
  }

  cnpy::npy_save("conv64x128x1x1.npy", output_matrix,
                 {1, NUM_FILTERS, (unsigned long)output_height,
                  (unsigned long)output_width},
                 "w");
  cout << "conv64x128x1x1.npy dumped" << endl;

  delete[] input;
  delete[] filters;
  delete[] bias;
  delete[] output_matrix;

  return 0;
}

/*
    g++ conv64x128x1x1.cc -o conv64x128x1x1  -std=c++11 -lcnpy
*/
