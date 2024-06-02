import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
from torch.nn import Conv2d


def conv4d(data, filters, bias=None, permute_filters=True, use_half=False):
    b, c, h, w, d, t = data.size()

    data = data.permute(
        2, 0, 1, 3, 4, 5
    ).contiguous()  # permute to avoid making contiguous inside loop

    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        filters = filters.permute(
            2, 0, 1, 3, 4, 5
        ).contiguous()  # permute to avoid making contiguous inside loop

    c_out = filters.size(1)
    if use_half:
        output = Variable(
            torch.HalfTensor(h, b, c_out, w, d, t), requires_grad=data.requires_grad
        )
    else:
        output = Variable(
            torch.zeros(h, b, c_out, w, d, t), requires_grad=data.requires_grad
        )

    padding = filters.size(0) // 2
    if use_half:
        Z = Variable(torch.zeros(padding, b, c, w, d, t).half())
    else:
        Z = Variable(torch.zeros(padding, b, c, w, d, t))

    if data.is_cuda:
        Z = Z.cuda(data.get_device())
        output = output.cuda(data.get_device())

    data_padded = torch.cat((Z, data, Z), 0)

    for i in range(output.size(0)):  # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i, :, :, :, :, :] = F.conv3d(
            data_padded[i + padding, :, :, :, :, :],
            filters[padding, :, :, :, :, :],
            bias=bias,
            stride=1,
            padding=padding,
        )
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1, padding + 1):
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(
                data_padded[i + padding - p, :, :, :, :, :],
                filters[padding - p, :, :, :, :, :],
                bias=None,
                stride=1,
                padding=padding,
            )
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(
                data_padded[i + padding + p, :, :, :, :, :],
                filters[padding + p, :, :, :, :, :],
                bias=None,
                stride=1,
                padding=padding,
            )

    output = output.permute(1, 2, 0, 3, 4, 5).contiguous()
    return output


class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        pre_permuted_filters=True,
    ):
        # stride, dilation and groups !=1 functionality not tested
        stride = 1
        dilation = 1
        groups = 1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _quadruple(0),
            groups,
            bias,
            padding_mode="zeros",
        )

        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor,
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters = pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data = self.weight.data.permute(2, 0, 1, 3, 4, 5).contiguous()
        self.use_half = False

    def forward(self, input):
        return conv4d(
            input,
            self.weight,
            bias=self.bias,
            permute_filters=not self.pre_permuted_filters,
            use_half=self.use_half,
        )  # filters pre-permuted in constructor
