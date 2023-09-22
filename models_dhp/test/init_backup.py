


from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet
from .unet_mod import  *

# def get_net( trial, input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
def get_net(trial):

    num_channels_down = [
            trial.suggest_int(name='num_channels_down_1', low=32, high=256),
            trial.suggest_int(name='num_channels_down_2', low=32, high=256),
            # Add more entries as needed
        ]
    num_channels_up = [
            trial.suggest_int(name='num_channels_up_1', low=32, high=256),
            trial.suggest_int(name='num_channels_up_2', low=32, high=256),
                           ]

    num_channels_skip = [
            trial.suggest_int(name='num_channels_skip_1', low=0, high=32),
            trial.suggest_int(name='num_channels_skip_2', low=0,high=32)
                             ]

    filter_size_down = [trial.suggest_int('filter_size_down_1', 1, 7),
                            trial.suggest_int('filter_size_down_2', 1, 7),
                            # ... Define other hyperparameters here
                            ]

    filter_size_up = [trial.suggest_int('filter_size_up_1', 1, 7),
                          trial.suggest_int('filter_size_up_2', 1, 7),
                          # ... Define other hyperparameters here
                          ]

    filter_skip_size = trial.suggest_int('filter_skip_size', 1, 3)

    need_sigmoid = trial.suggest_categorical('need_sigmoid', [True, False])
    need_bias = trial.suggest_categorical('need_bias', [True, False])
    pad = trial.suggest_categorical('pad', ['zero', 'reflection', 'replication', 'circular'])
    upsample_mode = trial.suggest_categorical('upsample_mode', ['nearest', 'bilinear'])
    downsample_mode = trial.suggest_categorical('downsample_mode', ['stride', 'avg', 'max', 'lanczos2'])
    act_fun = trial.suggest_categorical('act_fun', ['LeakyReLU', 'Swish', 'ELU', 'none'])
    need1x1_up = trial.suggest_categorical('need1x1_up', [True, False])

        # Create and train the model with the chosen hyperparameters
    net = skip(num_channels_down=num_channels_down,
                     num_channels_up=num_channels_up,
                     num_channels_skip=num_channels_skip,
                     filter_size_down=filter_size_down[0],
                     filter_size_up=filter_size_up[0],
                     filter_skip_size=filter_skip_size,
                     need_sigmoid=need_sigmoid,
                     need_bias=need_bias,
                     pad=pad,
                     upsample_mode=upsample_mode,
                     downsample_mode=downsample_mode,
                     act_fun=act_fun,
                     need1x1_up=need1x1_up)



    return net