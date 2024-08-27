#include <cnpy.h>

#include <fstream>
#include <iostream>

#define CHANNELS 128
#define HEIGHT 224
#define WIDTH 224
#define NUM_FILTERS 64
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define STRIDE 1
#define PADDING 0

using namespace std;

void conv3x3(float *input, float *output, float *filters, float *bias,
             int height, int width, int input_channels, int output_filters,
             int kernel_height, int kernel_width, int stride, int padding) {
  int padded_height = height + 2 * padding;
  int padded_width = width + 2 * padding;
  int output_height = (height + 2 * padding - kernel_height) / STRIDE + 1;
  int output_width = (width + 2 * padding - kernel_width) / STRIDE + 1;

  cout << "padded_height: " << padded_height
       << " padded_width: " << padded_width << endl;
  cout << "output_height: " << output_height
       << " output_width: " << output_width << endl;

  float *padded_input =
      new float[input_channels * padded_height * padded_width];

  // pad the input if PADDING is not 0
  for (int i = 0; i < input_channels; i++) {
    for (int j = 0; j < padded_height; j++) {
      for (int k = 0; k < padded_width; k++) {
        if (j < padding || j >= padded_height - padding || k < padding ||
            k >= padded_width - padding) {
          padded_input[i * padded_height * padded_width + j * padded_width +
                       k] = 0;
        } else {
          padded_input[i * padded_height * padded_width + j * padded_width +
                       k] =
              input[i * height * width + (j - padding) * width + (k - padding)];
        }
      }
    }
  }

  // Perform convolution
  for (int f = 0; f < output_filters; f++) {
    for (int i = 0; i < output_height; i++) {
      for (int j = 0; j < output_width; j++) {
        float sum = bias[f];
        for (int ch = 0; ch < input_channels; ch++) {
          for (int k = 0; k < kernel_height; k++) {
            for (int l = 0; l < kernel_width; l++) {
              int input_row = i * STRIDE + k;
              int input_col = j * STRIDE + l;
              sum += padded_input[ch * padded_height * padded_width +
                                  input_row * padded_width + input_col] *
                     filters[f * input_channels * kernel_height * kernel_width +
                             ch * kernel_height * kernel_width +
                             k * kernel_width + l];
            }
          }
        }
        output[f * output_height * output_width + i * output_width + j] = sum;
      }
    }
  }

  cnpy::npy_save("conv64x128x3x3.npy", output,
                 {1, NUM_FILTERS, (unsigned long)output_height,
                  (unsigned long)output_width},
                 "w");
  cout << "conv64x128x3x3.npy dumped" << endl;
}

int main() {
  //  input  : 1x128x224x224 (nchw)
  //  kernel : 64x128x3x3 (oihw)
  //  output : 1x64x222x222 [considering stride 1 and padding 0]

  float *input = new float[CHANNELS * HEIGHT * WIDTH];
  float *filters =
      new float[NUM_FILTERS * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH];
  float *bias = new float[NUM_FILTERS];
  // compute output size
  int output_height = (HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
  int output_width = (WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;
  float *output_matrix = new float[NUM_FILTERS * output_height * output_width];

  int num = 0;
  for (int i = 0; i < CHANNELS; i++) {
    for (int j = 0; j < HEIGHT; j++) {
      for (int k = 0; k < WIDTH; k++) {
        input[i * HEIGHT * WIDTH + (j)*WIDTH + (k)] = (++num);
      }
    }
  }
  num = 0;
  // Fill weights with 1 and biases with 0
  for (int f = 0; f < NUM_FILTERS; f++) {
    for (int ch = 0; ch < CHANNELS; ch++) {
      for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
          filters[f * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH +
                  ch * KERNEL_HEIGHT * KERNEL_WIDTH + i * KERNEL_WIDTH + j] =
              (++num);
        }
      }
    }
  }
  for (int i = 0; i < NUM_FILTERS; i++) {
    bias[i] = 0;
  }
  cout << "Initialization Done" << endl;

  conv3x3(input, output_matrix, filters, bias, HEIGHT, WIDTH, CHANNELS,
          NUM_FILTERS, KERNEL_HEIGHT, KERNEL_WIDTH, STRIDE, PADDING);

  delete[] input;
  delete[] filters;
  delete[] bias;
  delete[] output_matrix;

  return 0;
}

/*
    g++ conv64x128x3x3.cc -o conv64x128x3x3  -std=c++11 -lcnpy
*/
