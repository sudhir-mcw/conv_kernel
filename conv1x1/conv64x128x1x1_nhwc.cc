#include <cnpy.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>

#define CHANNELS 128
#define HEIGHT 224
#define WIDTH 224
#define NUM_FILTERS 64
#define KERNEL_HEIGHT 1
#define KERNEL_WIDTH 1
#define STRIDE 1
#define PADDING 1

using namespace std;

void conv1x1(float ***input_nhwc, float ***output_nhwc, float ****filters,
             float *bias, int height, int width, int input_channels,
             int output_filters, int kernel_height, int kernel_width,
             int stride, int padding) {
  int padded_height = height + 2 * padding;
  int padded_width = width + 2 * padding;
  int output_height = ((height + 2 * padding - kernel_height) / stride) + 1;
  int output_width = ((width + 2 * padding - kernel_width) / stride) + 1;

  float ***padded_input_nhwc = new float **[padded_height];
  for (int i = 0; i < padded_height; i++) {
    padded_input_nhwc[i] = new float *[padded_width];
    for (int j = 0; j < padded_width; j++) {
      padded_input_nhwc[i][j] = new float[input_channels];
    }
  }
  for (int i = 0; i < padded_height; i++) {
    for (int j = 0; j < padded_width; j++) {
      for (int ch = 0; ch < input_channels; ch++) {
        if (i < padding || i >= padded_height - padding || j < padding ||
            j >= padded_width - padding) {
          padded_input_nhwc[i][j][ch] = 0;

        } else {
          padded_input_nhwc[i][j][ch] =
              input_nhwc[i - padding][j - padding][ch];
        }
      }
    }
  }
  cout << "padding done" << endl;

  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      for (int k = 0; k < output_filters; k++) {
        output_nhwc[i][j][k] = bias[k];
        for (int l = 0; l < kernel_height; l++) {
          for (int m = 0; m < kernel_width; m++) {
            for (int ch = 0; ch < input_channels; ch++) {
              int input_row = i * STRIDE + l;
              int input_col = j * STRIDE + m;
              output_nhwc[i][j][k] +=
                  (padded_input_nhwc[input_row][input_col][ch]) *
                  (filters[l][m][ch][k]);
            }
          }
        }
      }
    }
  }
  cout << "conv1x1 done : [" << "1," << output_height << "," << output_width
       << "," << output_filters << "]" << endl;

  float *flattend_output =
      new float[output_height * output_width * output_filters];
  int offset = 0;
  for (int i = 0; i < output_height; i++) {
    for (int j = 0; j < output_width; j++) {
      std::memcpy(flattend_output + offset, output_nhwc[i][j],
                  sizeof(float) * output_filters);
      offset += output_filters;
    }
  }

  cnpy::npy_save("conv64x128x1x1_nhwc.npy", flattend_output,
                 {1, (unsigned long)output_height, (unsigned long)output_width,
                  NUM_FILTERS},
                 "w");
  std::cout << "conv64x128x1x1_nchw.npy dumped" << endl;
}

int main() {
  //  input  : 1x128x224x224 (nchw)
  //  kernel : 64x128x1x1 (oihw)
  //  output : 1x64x224x224 [considering stride 1 and padding 0]
  float ***nhwc_input = new float **[HEIGHT];
  for (int i = 0; i < HEIGHT; i++) {
    nhwc_input[i] = new float *[WIDTH];
    for (int j = 0; j < WIDTH; j++) {
      nhwc_input[i][j] = new float[CHANNELS];
    }
  }
  float ****nhwc_filters = new float ***[KERNEL_HEIGHT];
  for (int i = 0; i < KERNEL_HEIGHT; i++) {
    nhwc_filters[i] = new float **[KERNEL_WIDTH];
    for (int j = 0; j < KERNEL_WIDTH; j++) {
      nhwc_filters[i][j] = new float *[CHANNELS];
      for (int k = 0; k < CHANNELS; k++) {
        nhwc_filters[i][j][k] = new float[NUM_FILTERS];
      }
    }
  }

  float *bias = new float[NUM_FILTERS];
  // compute output size
  int output_height = (HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
  int output_width = (WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;
  float ***output_matrix = new float **[output_height];
  for (int i = 0; i < output_height; i++) {
    output_matrix[i] = new float *[output_width];
    for (int j = 0; j < output_width; j++) {
      output_matrix[i][j] = new float[NUM_FILTERS];
    }
  }

  int num = 0;
  for (int i = 0; i < CHANNELS; i++) {
    for (int j = 0; j < HEIGHT; j++) {
      for (int k = 0; k < WIDTH; k++) {
        nhwc_input[j][k][i] = (++num);
      }
    }
  }
  num = 0;
  // Fill weights with 1 and biases with 0
  for (int f = 0; f < NUM_FILTERS; f++) {
    for (int ch = 0; ch < CHANNELS; ch++) {
      for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
          nhwc_filters[i][j][ch][f] = (++num);
        }
      }
    }
  }
  for (int i = 0; i < NUM_FILTERS; i++) {
    bias[i] = 0;
  }
  std::cout << "Initialization Done" << endl;
  conv1x1(nhwc_input, output_matrix, nhwc_filters, bias, HEIGHT, WIDTH,
          CHANNELS, NUM_FILTERS, KERNEL_HEIGHT, KERNEL_WIDTH, STRIDE, PADDING);

  return 0;
}

/*
    g++ conv64x128x1x1.cc -o conv64x128x1x1  -std=c++11 -lcnpy
*/
