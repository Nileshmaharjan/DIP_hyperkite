# import torch
# import numpy as np
#
#
# def get_noise(input_depth, method, spatial_size, poisson_var=None, gaussian_var=None):
#     """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
#     initialized with Poisson and Gaussian noise added to it.
#     Args:
#         input_depth: number of channels in the tensor
#         method: `noise` for filling tensor with noise; `meshgrid` for np.meshgrid
#         spatial_size: spatial size of the tensor to initialize
#         poisson_var: a factor for Poisson noise (None if no Poisson noise is desired)
#         gaussian_var: a factor for Gaussian noise (None if no Gaussian noise is desired)
#     """
#     if isinstance(spatial_size, int):
#         spatial_size = (spatial_size, spatial_size)
#     shape = [1, input_depth, spatial_size[0], spatial_size[1]]
#     net_input = torch.zeros(shape)
#
#     if poisson_var is not None:
#         fill_noise(net_input, 'p', poisson_var)  # Add Poisson noise
#
#     if gaussian_var is not None:
#         fill_noise(net_input, 'g', gaussian_var)  # Add Gaussian noise
#
#     if method == 'meshgrid':
#         assert input_depth == 2
#         X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
#                            np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
#         meshgrid = np.concatenate([X[None, :], Y[None, :]])
#         net_input = np_to_torch(meshgrid)
#
#     return net_input
#
#
# def fill_noise(noise, noise_type, var):
#     """Fill the input tensor `noise` with different types of noise.
#     Args:
#         noise: Input tensor to be filled with noise.
#         noise_type: Type of noise ('u' for uniform, 'n' for normal, 'p' for Poisson, 'g' for Gaussian).
#         var: Standard deviation for Gaussian noise or mean for Poisson noise.
#     """
#     if noise_type == 'u':  # Uniform noise
#         noise.uniform_(-var, var)
#     elif noise_type == 'n':  # Normal (Gaussian) noise
#         noise.normal_(0, var)
#     elif noise_type == 'p':  # Poisson noise
#         noise.poisson_(var)
#     elif noise_type == 'g':  # Gaussian noise with mean 'var'
#         noise.normal_(var, var)
#     else:
#         raise ValueError(
#             "Unsupported noise_type. Use 'u' for uniform, 'n' for normal, 'p' for Poisson, 'g' for Gaussian.")
#
#
# def np_to_torch(np_array):
#     """Convert a NumPy array to a PyTorch tensor."""
#     return torch.from_numpy(np_array).float()
