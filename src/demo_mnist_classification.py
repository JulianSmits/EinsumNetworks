import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork, FactorizedLeafLayer
from EinsumNetwork.EinetMixture import EinetMixture
import datasets
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

demo_text = """
This demo loads (fashion) mnist and quickly trains an EiNet for some epochs. 

There are some parameters to play with, as for example which exponential family you want 
to use, which classes you want to pick, and structural parameters. Then an EiNet is trained, 
the log-likelihoods reported, some (conditional and unconditional) samples are produced, and
approximate MPE reconstructions are generated. 
"""
print(demo_text)

############################################################################
fashion_mnist = False

exponential_family = EinsumNetwork.BinomialArray
# exponential_family = EinsumNetwork.CategoricalArray
# exponential_family = EinsumNetwork.NormalArray

classes = [2, 4]
# classes = [2, 3, 5, 7]
# classes = None

K = 10

structure = 'poon-domingos'
# structure = 'binary-trees'

# 'poon-domingos'
pd_num_pieces = [4]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 28
height = 28

# 'binary-trees'
depth = 3
num_repetitions = 20

num_epochs = 20
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

use_custom_initializer = False
############################################################################

exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

# get data
if fashion_mnist:
    train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
else:
    train_x, train_labels, test_x, test_labels = datasets.load_mnist()

if not exponential_family != EinsumNetwork.NormalArray:
    train_x /= 255.
    test_x /= 255.
    train_x -= .5
    test_x -= .5

# validation split
valid_x = train_x[-10000:, :]
train_x = train_x[:-10000, :]
valid_labels = train_labels[-10000:]
train_labels = train_labels[:-10000]
# pick the selected classes
if classes is not None:
    train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
    valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
    test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

    train_labels = [l for l in train_labels if l in classes]
    valid_labels = [l for l in valid_labels if l in classes]
    test_labels = [l for l in test_labels if l in classes]
else:
    classes = np.unique(train_labels)

train_x = torch.from_numpy(train_x).to(torch.device(device))
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))

# Make EinsumNetworks for each class
######################################
def custom_initializer(layer, train_x):
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


einets = []
p = []
for c in classes:
    if structure == 'poon-domingos':
        pd_delta = [[height / d, width / d] for d in pd_num_pieces]
        graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
    elif structure == 'binary-trees':
        graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
    else:
        raise AssertionError("Unknown Structure")

    args = EinsumNetwork.Args(
            num_var=train_x.shape[1],
            num_dims=1,
            num_classes=1,
            num_sums=K,
            num_input_distributions=K,
            exponential_family=exponential_family,
            exponential_family_args=exponential_family_args,
            online_em_frequency=online_em_frequency,
            online_em_stepsize=online_em_stepsize)

    einet = EinsumNetwork.EinsumNetwork(graph, args)

    init_dict = None
    if use_custom_initializer:
        init_dict = {}
        for layer in einet.einet_layers:
            init_dict[layer] = custom_initializer(layer, train_x)

    einet.initialize(init_dict)
    einet.to(device)
    print(einet)
    einets.append(einet)

# Train
######################################

for (einet, c) in zip(einets, classes):
  train_x_c = train_x[[l == c for l in train_labels]]
  valid_x_c = valid_x[[l == c for l in valid_labels]]
  test_x_c = test_x[[l == c for l in test_labels]]

  train_N = train_x_c.shape[0]
  valid_N = valid_x_c.shape[0]
  test_N = test_x_c.shape[0]

  p.append(train_N)

  for epoch_count in range(num_epochs):

      ##### evaluate
      einet.eval()
      train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x_c, batch_size=batch_size)
      valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x_c, batch_size=batch_size)
      test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x_c, batch_size=batch_size)
      print("[{}]   train LL {}   valid LL {}   test LL {}".format(
          epoch_count,
          train_ll / train_N,
          valid_ll / valid_N,
          test_ll / test_N))
      einet.train()
      #####

      idx_batches = torch.randperm(train_N, device=device).split(batch_size)

      total_ll = 0.0
      for idx in idx_batches:
          batch_x = train_x_c[idx, :]
          outputs = einet.forward(batch_x)
          ll_sample = EinsumNetwork.log_likelihoods(outputs)
          log_likelihood = ll_sample.sum()
          log_likelihood.backward()

          einet.em_process_batch()
          total_ll += log_likelihood.detach().item()

      einet.em_update()

if fashion_mnist:
    model_dir = '../models/einet/demo_fashion_mnist/'
    samples_dir = '../samples/demo_fashion_mnist/'
else:
    model_dir = '../models/einet/demo_mnist/'
    samples_dir = '../samples/demo_mnist/'
utils.mkdir_p(model_dir)
utils.mkdir_p(samples_dir)


################################################
# construct EinetMixture and do classification #
################################################

print("-------- classification ----------")

mixture = EinetMixture(p, einets, classes=classes)

sample_idx = [0, 10, 20, 30]
samples = test_x[sample_idx]

correct_labels = [test_labels[i] for i in sample_idx]
predictions = mixture.classify_samples(samples)

for (i, c, p) in zip(sample_idx, correct_labels, predictions):
    print(f'test index {i}: correct label = {c}, predicted label = {p}')

for (einet, c) in zip(einets, classes):
    samples = einet.sample(num_samples=25).cpu().numpy()
    samples = samples.reshape((-1, 28, 28))
    utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, f"samples{c}.png"), margin_gray_val=0.)

####################
# save and re-load #
####################

# # evaluate log-likelihoods
# einet.eval()
# train_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
# valid_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
# test_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)

# # save model
# graph_file = os.path.join(model_dir, "einet.pc")
# Graph.write_gpickle(graph, graph_file)
# print("Saved PC graph to {}".format(graph_file))
# model_file = os.path.join(model_dir, "einet.mdl")
# torch.save(einet, model_file)
# print("Saved model to {}".format(model_file))

# del einet

# # reload model
# einet = torch.load(model_file)
# print("Loaded model from {}".format(model_file))

# # evaluate log-likelihoods on re-loaded model
# train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
# valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
# test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
# print()
# print("Log-likelihoods before saving --- train LL {}   valid LL {}   test LL {}".format(
#         train_ll / train_N,
#         valid_ll / valid_N,
#         test_ll / test_N))
# print("Log-likelihoods after saving  --- train LL {}   valid LL {}   test LL {}".format(
#         train_ll / train_N,
#         valid_ll / valid_N,
#         test_ll / test_N))
