# from .skip import skip
# from .texture_nets import get_texture_nets
# from .resnet import ResNet
# from .unet import UNet
# from .unet_mod import *

#
# def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
#     if NET_TYPE == 'ResNet':
#         print('here')
#         # net = ResNet(
#         #     num_input_channels=input_depth, num_output_channels=102,
#         #     num_blocks=10, num_channels=16,
#         #     need_residual=True, act_fun='LeakyReLU', need_sigmoid=True, norm_layer=nn.BatchNorm2d, pad='reflection')
#     elif NET_TYPE == 'skip':
#         net = skip(input_depth, n_channels,
#                    num_channels_down=[skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d,
#                    num_channels_up=[skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u,
#                    num_channels_skip=[skip_n11] * num_scales if isinstance(skip_n11, int) else skip_n11,
#                    filter_size_down=3, filter_size_up=3, filter_skip_size=1,
#                    upsample_mode=upsample_mode, downsample_mode=downsample_mode,
#                    need_sigmoid=False, need_bias=True, pad=pad, act_fun=act_fun)
#         print('here')
#         # input_depth = 32,
#         # n_channel = 102,
#         # skip_n33u = 128,
#         # skip_n11  = 4,
#         # num_scales = 5,
#         # skip_n33d = 128,
#         # Define hyperparameters to optimize
#
#     elif NET_TYPE == 'texture_nets':
#         net = get_texture_nets(inp=input_depth, ratios = [32, 16, 8, 4, 2, 1], fill_noise=False,pad=pad)
#
#     elif NET_TYPE =='UNet':
#         net = UNet(num_input_channels=input_depth, num_output_channels=102,
#                    feature_scale=4, more_layers=0, concat_x=False,
#                    upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)
#         print('net', net)
#     elif NET_TYPE == 'identity':
#         assert input_depth == 3
#         net = nn.Sequential()
#     elif NET_TYPE == 'UNet_mod':
#         net = UNet_mod(in_nc=input_depth, out_nc=102, nc=[64, 128, 256, 512],)
#     elif NET_TYPE == 'UNet_res':
#         net = UNetRes(in_nc=input_depth, out_nc=102, nc=[64, 128, 256, 512],)
#     elif NET_TYPE == 'ResUNet':
#         net = ResUNet(in_nc=input_depth, out_nc=102, nc=[64, 128, 256, 512],)
#     else:
#         assert False
#
#
#     return net
#

# For NAS

from models_dhp.nas.DIP.models.skip import skip

def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    if NET_TYPE == 'ResNet':
        print('here')
        # net = ResNet(
        #     num_input_channels=input_depth, num_output_channels=102,
        #     num_blocks=10, num_channels=16,
        #     need_residual=True, act_fun='LeakyReLU', need_sigmoid=True, norm_layer=nn.BatchNorm2d, pad='reflection')
    elif NET_TYPE == 'skip':
        net = skip(
                   num_input_channels=32,
                   num_output_channels = 102,
                   num_channels_down=[skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d,
                   num_channels_up=[skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u,
                   num_channels_skip=[skip_n11] * num_scales if isinstance(skip_n11, int) else skip_n11,
                   filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                   upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                   need_sigmoid=False, need_bias=True, pad=pad, act_fun=act_fun)

    else:
        assert False


    return net
