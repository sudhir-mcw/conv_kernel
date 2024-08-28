from torch import nn
import torch
import numpy as np

INPUT_CHANNELS = 128
OUTPUT_CHANNELS = 64
KERNEL_SIZE = 1
STRIDE = 1
PADDING = 1
HEIGHT = 224
WIDTH = 224

if __name__ == "__main__":

    conv = nn.Conv2d(
        OUTPUT_CHANNELS,
        INPUT_CHANNELS,
        kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
        stride=STRIDE,
        padding=PADDING,
    )

    input_matrix = [(i + 1) for i in range(0, (INPUT_CHANNELS * HEIGHT * WIDTH))]
    input_tensor = torch.Tensor(input_matrix).reshape(1, INPUT_CHANNELS, HEIGHT, WIDTH)
    print("input shape ", input_tensor.shape)

    filters_matrix = [
        (i + 1)
        for i in range(
            0, (OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE)
        )
    ]
    filters_tensor = torch.Tensor(filters_matrix).reshape(
        OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE
    )
    print("weights shape ", filters_tensor.shape)

    bias_matrix = [(0) for i in range(OUTPUT_CHANNELS)]
    bias_tensor = torch.Tensor(bias_matrix)

    conv.weight.data = filters_tensor
    conv.bias.data = bias_tensor

    output = conv(input_tensor)
    print("output shape ", output.shape)
    np.save("py_conv64x128x1x1.npy", output.detach().numpy())
