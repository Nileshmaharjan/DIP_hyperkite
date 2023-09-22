import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *
#
def skip(
        num_input_channels=32, num_output_channels=102,
        num_channels_down=[128, 128, 128, 128, 128], num_channels_up=[128, 128, 128, 128, 128], num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=False, need_bias=True,
        pad='reflection', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    # checks that the number of channels specified for downscaling, upscaling, and skip connections is the same.
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    # Calculates the number of scales based on the length of the "num_channels_down" list
    n_scales = len(num_channels_down)

    # Lines ensure that the "upsample_mode" and "downsample_mode" arguments
    # are either lists or tuples, and if they are not, they convert them into
    # lists with the same value repeated for each scale.

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales

    # nes ensure that the filter sizes for downscaling and upscaling
    # are lists or tuples. If not, they are converted into lists with
    # the same value repeated for each scale.

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:

            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))

            skip.add(bn(num_channels_skip[i]))

            skip.add(act(act_fun))


        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))

        deeper.add(bn(num_channels_down[i]))

        deeper.add(act(act_fun))


        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))

        deeper.add(bn(num_channels_down[i]))

        deeper.add(act(act_fun))


        deeper_main = nn.Sequential()


        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]

        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]


        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))


        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))

        model_tmp.add(bn(num_channels_up[i]))

        model_tmp.add(act(act_fun))



        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        # deeper ko value deeper_main ma janchha
        model_tmp = deeper_main


    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a dilated convolution block
def dilated_conv(in_channels, out_channels, kernel_size, stride, dilation, bias=True, pad='reflection'):
    if pad == 'reflection':
        padding = (dilation * (kernel_size - 1)) // 2
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=bias)
    elif pad == 'zero':
        padding = dilation
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=bias)
    else:
        raise ValueError("Unsupported padding type: {}".format(pad))

    return conv

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute query, key, and value
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # Reshape query, key, and value for batch matrix multiplication
        proj_query = proj_query.view(batch_size, -1, height * width)
        proj_key = proj_key.view(batch_size, -1, height * width)
        proj_value = proj_value.view(batch_size, -1, height * width)  # Reduce channels here

        proj_query = proj_query.permute(0, 2, 1)  # Swap dimensions 1 and 2
        proj_value = proj_value.permute(0, 2, 1)
        # Scaled dot-product attention
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy / (height * width) ** 0.5, dim=-1)  # Proper scaling
        out = torch.bmm(attention, proj_value)  # Updated attention dimension
        out = out.view(batch_size, channels, height, width)  # Reshape back to the original shape

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias=True, pad='reflection'):
        super(ResidualBlock, self).__init__()
        self.conv1 = dilated_conv(in_channels, out_channels, kernel_size, stride, dilation, bias=bias, pad=pad)
        self.relu = nn.ReLU(inplace=True)
        self.attention = SelfAttention(out_channels)
        self.upsample_out = nn.Upsample(scale_factor=2, mode='bilinear')

        # Adjust the number of channels in residual if needed
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.attention(out)

        # Upsample 'out' to match the spatial dimensions of 'residual'
        out = self.upsample_out(out)

        # If the number of channels in residual is different, apply convolution
        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(residual)

        out += residual  # Add residual connection
        return out


def skip(
    num_input_channels=32, num_output_channels=102,
    num_channels_down=[128, 128, 128, 128, 128], num_channels_up=[128, 128, 128, 128, 128],
    num_channels_skip=[4, 4, 4, 4, 4],
    filter_size_down=3, filter_size_up=3, filter_skip_size=1,
    need_sigmoid=False, need_bias=True,
    pad='reflection', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU',
    need1x1_up=True, dilations=None):  # Add dilations parameter
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)

    if not isinstance(upsample_mode, list) or not isinstance(upsample_mode, tuple):
        upsample_mode = [upsample_mode] * n_scales

    if not isinstance(downsample_mode, list) or not isinstance(downsample_mode, tuple):
        downsample_mode = [downsample_mode] * n_scales

    if not isinstance(filter_size_down, list) or not isinstance(filter_size_down, tuple):
        filter_size_down = [filter_size_down] * n_scales

    if not isinstance(filter_size_up, list) or not isinstance(filter_size_up, tuple):
        filter_size_up = [filter_size_up] * n_scales

    if dilations is None:
        dilations = [1] * len(num_channels_down)
    else:
        assert len(dilations) == len(num_channels_down)

    last_scale = n_scales - 1
    cur_depth = None
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels

    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        deeper.add(ResidualBlock(input_depth, num_channels_down[i], filter_size_down[i], 2, dilations[i], bias=need_bias, pad=pad))  # Use dilated convolution

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))

    if need_sigmoid:
        model.add(nn.Sigmoid())
    print('here')

    return model
