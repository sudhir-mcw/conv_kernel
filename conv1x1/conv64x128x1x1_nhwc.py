import numpy as np
import torch
from torch import nn


CHANNELS = 128
HEIGHT = 224
WIDTH = 224
KERNEL_HEIGHT = 1
KERNEL_WIDTH = 1
OUTPUT_CHANNELS = 64
STRIDE = 2
PADDING = 2

if __name__ == "__main__":

    input_matrix = [(i + 1) for i in range(0, (1 * CHANNELS * HEIGHT * WIDTH))]
    input_tensor = torch.Tensor(input_matrix).reshape(1, CHANNELS, HEIGHT, WIDTH)

    input_tensor_nhwc = input_tensor.permute(0, 2, 3, 1)  # for nhwc conversion
    print("nchw input shape ", input_tensor.shape," nhwc input shape ", input_tensor_nhwc.shape)
    np.save("py_input_nhwc.npy", input_tensor_nhwc.detach().numpy())

    weight_matrix = [
        (i + 1)
        for i in range(
            0, (1 * OUTPUT_CHANNELS * CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH)
        )
    ]
    weight_tensor = torch.Tensor(weight_matrix).reshape(
        OUTPUT_CHANNELS, CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH
    )

    weight_tensor_nhwc = weight_tensor.permute(2, 3, 1, 0)  # for nhwc conversion
    print("nchw weight shape ", weight_tensor.shape," nhwc weight shape ", weight_tensor_nhwc.shape)
    np.save("py_weight_nhwc.npy", weight_tensor_nhwc.detach().numpy())

    bias_matrix = [(0) for i in range(0, (1 * OUTPUT_CHANNELS))]
    bias_tensor = torch.Tensor(bias_matrix)

    conv = nn.Conv2d(
        OUTPUT_CHANNELS,
        CHANNELS,
        kernel_size=(KERNEL_HEIGHT, KERNEL_WIDTH),
        stride=STRIDE,
        padding=PADDING,
    )
    conv.weight.data = weight_tensor
    conv.bias.data = bias_tensor

    output_tensor = conv(input_tensor)

    output_tensor_nhwc = output_tensor.permute(0, 2, 3, 1)  # for nhwc conversion

    print("nchw output shape ", output_tensor.shape," nhwc output shape ", output_tensor_nhwc.shape)
    np.save("py_conv64x128x1x1_nhwc.npy", output_tensor_nhwc.detach().numpy())
