#include <cnpy.h>

#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>

#define CHANNELS 128
#define HEIGHT 224
#define WIDTH 224
#define NUM_FILTERS 64
#define KERNEL_HEIGHT 1
#define KERNEL_WIDTH 1
#define STRIDE 1
#define PADDING 0

using namespace std;

void conv1x1(float ***input, float ***output, float ****filters, float *bias,
             int height, int width, int input_channels, int output_filters,
             int kernel_height, int kernel_width, int stride, int padding)
{
  int padded_height = height + 2 * padding;
  int padded_width = width + 2 * padding;
  int output_height = (height + 2 * padding - kernel_height) / STRIDE + 1;
  int output_width = (width + 2 * padding - kernel_width) / STRIDE + 1;

  std::cout << "padded_height: " << padded_height
            << " padded_width: " << padded_width << endl;
  std::cout << "output_height: " << output_height
            << " output_width: " << output_width << endl;

  float ***padded_input = new float **[input_channels];
  for (int i = 0; i < input_channels; i++)
  {
    padded_input[i] = new float *[padded_height];
    for (int j = 0; j < padded_height; j++)
    {
      padded_input[i][j] = new float[padded_width];
    }
  }

  // pad the input if PADDING is not 0
  for (int i = 0; i < input_channels; i++)
  {
    for (int j = 0; j < padded_height; j++)
    {
      for (int k = 0; k < padded_width; k++)
      {
        if (j < padding || j >= padded_height - padding || k < padding ||
            k >= padded_width - padding)
        {
          padded_input[i][j][k] = 0;
        }
        else
        {
          padded_input[i][j][k] = input[i][j - padding][k - padding];
        }
      }
    }
  }

  // Perform convolution
  for (int f = 0; f < output_filters; f++)
  {
    for (int i = 0; i < output_height; i++)
    {
      for (int j = 0; j < output_width; j++)
      {
        float sum = bias[f];
        for (int ch = 0; ch < input_channels; ch++)
        {
          for (int k = 0; k < kernel_height; k++)
          {
            for (int l = 0; l < kernel_width; l++)
            {
              int input_row = i * STRIDE + k;
              int input_col = j * STRIDE + l;
              sum += (padded_input[ch][input_row][input_col]) * (filters[f][ch][k][l]);
            }
          }
        }
        output[f][i][j] = sum;
      }
    }
  }

  float *flattend_output = new float[output_filters * output_height * output_width];
  int offset = 0;
  for (int i = 0; i < output_filters; i++)
  {
    for (int j = 0; j < output_height; j++)
    {
      std::memcpy(flattend_output + offset, output[i][j], sizeof(float) * output_width);
      offset += output_width;
    }
  }

  cnpy::npy_save("conv64x128x1x1.npy", flattend_output,
                 {1, NUM_FILTERS, (unsigned long)output_height,
                  (unsigned long)output_width},
                 "w");
  std::cout << "conv64x128x1x1.npy dumped" << endl;
}

int main()
{
  //  input  : 1x128x224x224 (nchw)
  //  kernel : 64x128x1x1 (oihw)
  //  output : 1x64x224x224 [considering stride 1 and padding 0]

  float ***input = new float **[CHANNELS];
  for (int i = 0; i < CHANNELS; i++)
  {
    input[i] = new float *[HEIGHT];
    for (int j = 0; j < HEIGHT; j++)
    {
      input[i][j] = new float[WIDTH];
    }
  }

  float ****filters = new float ***[NUM_FILTERS];
  for (int i = 0; i < NUM_FILTERS; i++)
  {
    filters[i] = new float **[CHANNELS];
    for (int j = 0; j < CHANNELS; j++)
    {
      filters[i][j] = new float *[KERNEL_HEIGHT];
      for (int k = 0; k < KERNEL_HEIGHT; k++)
      {
        filters[i][j][k] = new float[KERNEL_WIDTH];
      }
    }
  }

  float *bias = new float[NUM_FILTERS];
  // compute output size
  int output_height = (HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
  int output_width = (WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;
  float ***output_matrix = new float **[NUM_FILTERS];
  for (int i = 0; i < NUM_FILTERS; i++)
  {
    output_matrix[i] = new float *[output_height];
    for (int j = 0; j < output_height; j++)
    {
      output_matrix[i][j] = new float[output_width];
    }
  }

  int num = 0;
  for (int i = 0; i < CHANNELS; i++)
  {
    for (int j = 0; j < HEIGHT; j++)
    {
      for (int k = 0; k < WIDTH; k++)
      {
        input[i][j][k] = (++num);
      }
    }
  }
  num = 0;
  // Fill weights with 1 and biases with 0
  for (int f = 0; f < NUM_FILTERS; f++)
  {
    for (int ch = 0; ch < CHANNELS; ch++)
    {
      for (int i = 0; i < KERNEL_HEIGHT; i++)
      {
        for (int j = 0; j < KERNEL_WIDTH; j++)
        {
          filters[f][ch][i][j] = (++num);
        }
      }
    }
  }
  for (int i = 0; i < NUM_FILTERS; i++)
  {
    bias[i] = 0;
  }
  std::cout << "Initialization Done" << endl;

  conv1x1(input, output_matrix, filters, bias, HEIGHT, WIDTH, CHANNELS,
          NUM_FILTERS, KERNEL_HEIGHT, KERNEL_WIDTH, STRIDE, PADDING);

  for (int i = 0; i < CHANNELS; i++)
  {
    for (int j = 0; j < HEIGHT; j++)
    {
      delete[] input[i][j];
    }
    delete[] input[i];
  }
  for (int i = 0; i < NUM_FILTERS; i++)
  {
    for (int j = 0; j < CHANNELS; j++)
    {
      for (int k = 0; k < KERNEL_HEIGHT; k++)
      {
        delete[] filters[i][j][k];
      }
      delete[] filters[i][j];
    }
    delete[] filters[i];
  }
  for (int i = 0; i < NUM_FILTERS; i++)
  {
    for (int j = 0; j < output_height; j++)
    {
      delete[] output_matrix[i][j];
    }
    delete[] output_matrix[i];
  }

  delete[] input;
  delete[] filters;
  delete[] bias;
  delete[] output_matrix;

  return 0;
}

/*
    g++ conv64x128x1x1.cc -o conv64x128x1x1  -std=c++11 -lcnpy
*/
