
import torch
from EinsumNetwork import FactorizedLeafLayer

use_custom_initializer = True

standard_random_initializer = True
gaussian_random_initializer = True


def get_init_dict(einet, train_x, train_labels=None, einet_class=None):
    """
    returns init_dict for einet
    """
    if einet is None:
        return

    # filter training samples when needed
    if einet_class is not None and train_labels is not None:
        train_x = train_x[[l == einet_class for l in train_labels]]


    # specify init_func based on settings
    init_func = None
    init_text = None
    if standard_random_initializer:
        init_func = standard_random_initializer
        init_text = """Initializer function: Standard_random_initializer
        """
    elif standard_random_initializer:
        init_func = gaussian_random_initializer
        init_text = """Initializer function: gaussian_random_initializer
        """
    print(init_text)

    # when no custom initilizer function is specified, return None
    if init_func is None:
        return None

    # fill init_dict with correct initializations
    init_dict = {}
    for layer in einet.einet_layers:
        init_dict[layer] = init_func(layer)

    return init_dict

def gaussian_random_initializer(layer):
    """
    Returns a tensor of random numbers drawn from separate normal distributions with mean 0.5 and standard deviation 1/6 clamped between 0 and 1.
    For leaf instances this is multiplied by N
    """
    mean = 0.5
    std = 1/6

    if isinstance(layer, FactorizedLeafLayer.FactorizedLeafLayer):
        ef_array = layer.ef_array
        m = mean * torch.ones(ef_array.num_var, *ef_array.array_shape, ef_array.num_dims)
        s = std * torch.ones(ef_array.num_var, *ef_array.array_shape, ef_array.num_dims)
        phi = (0.01 + 0.98 * torch.clamp(torch.normal(m, s), 0, 1)) * ef_array.N
        return phi

    m = mean * torch.ones(layer.params_shape)
    s = std * torch.ones(layer.params_shape)
    params = 0.01 + 0.98 * torch.clamp(torch.normal(m, s), 0, 1)
    with torch.no_grad():
        if layer.params_mask is not None:
            params.data *= layer.params_mask
        params.data = params.data / (params.data.sum(layer.normalization_dims, keepdim=True))
    return params
    pass

def standard_random_initializer(layer):
    """
    Returns a tensor filled with random numbers from a uniform distribution on the interval [0.01, 0.99)
    For leaf instances this is multiplied by N
    """
    if isinstance(layer, FactorizedLeafLayer.FactorizedLeafLayer):
        ef_array = layer.ef_array
        phi = (0.01 + 0.98 * torch.rand(ef_array.num_var, *ef_array.array_shape, ef_array.num_dims)) * ef_array.N
        return phi

    params = 0.01 + 0.98 * torch.rand(layer.params_shape)
    with torch.no_grad():
        if layer.params_mask is not None:
            params.data *= layer.params_mask
        params.data = params.data / (params.data.sum(layer.normalization_dims, keepdim=True))
    return params

def clustor_initializer(layer, train_x):
    """
    Initialize weights of the binomials with the average pixel values in the input dataset
    """
    if isinstance(layer, FactorizedLeafLayer.FactorizedLeafLayer):
        ef_array = layer.ef_array

        num_var = ef_array.num_var
        array_shape = ef_array.array_shape
        num_dims = ef_array.num_dims
        N = ef_array.N

        norm_phi = torch.einsum('ij->j', train_x) / (train_x.shape[0] * N)

        """ 
        random values from -0.05 to 0.05, -0.1 to 0.1 and -0.2 to 0.2 respectivly 
        does not provide results when only initializing Leaf Layer
        """
        # rand_val = torch.rand(norm_phi.shape).to(torch.device(device)) * 0.1 - 0.05 
        # rand_val = torch.rand(norm_phi.shape).to(torch.device(device)) * 0.2 - 0.1 
        # rand_val = torch.rand(norm_phi.shape).to(torch.device(device)) * 0.4 - 0.2 
        # norm_phi = torch.clamp(norm_phi.add(rand_val), 0, 1)

        phi_tensor = norm_phi.repeat(*array_shape, num_dims, 1).permute(3, 0, 1, 2)

        return phi_tensor * N
    
    """
    A simple initializer for normalized sum-weights.
    :return: initial parameters
    """
    params = 0.01 + 0.98 * torch.rand(layer.params_shape)
    with torch.no_grad():
        if layer.params_mask is not None:
            params.data *= layer.params_mask
        params.data = params.data / (params.data.sum(layer.normalization_dims, keepdim=True))
    return params
