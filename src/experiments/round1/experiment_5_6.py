import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork, FactorizedLeafLayer
from EinsumNetwork.EinetMixture import EinetMixture
import datasets
import utils
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

demo_text = """
This demo loads (fashion) mnist and quickly trains an EiNet for some epochs. 

There are some parameters to play with, as for example which exponential family you want 
to use, which classes you want to pick, and structural parameters. Then an EiNet is trained, 
the log-likelihoods reported, some (conditional and unconditional) samples are produced, and
approximate MPE reconstructions are generated. 

Experiment 5:
A single Einet is learned by executing a SGD algorithm with an Cross Entropy objective function. The log likelihood is measured.

Experiment 6:
A single Einet is learned by executing a SGD algorithm with an Cross Entropy objective function.  A classification task is executed.
"""
print(demo_text)

############################################################################
fashion_mnist = False

exponential_family = EinsumNetwork.BinomialArray
# exponential_family = EinsumNetwork.CategoricalArray
# exponential_family = EinsumNetwork.NormalArray

# classes = [7]
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

num_epochs = 5
batch_size = 100
SGD_learning_rate = 0.1

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
    classes = np.unique(train_labels).tolist()

    train_labels = [l for l in train_labels if l in classes]
    valid_labels = [l for l in valid_labels if l in classes]
    test_labels = [l for l in test_labels if l in classes]

train_x = torch.from_numpy(train_x).to(torch.device(device))
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))

######################################
# Make EinsumNetworks for each class #
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
        num_classes=len(classes),
        num_sums=K,
        num_input_distributions=K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        use_em=False)

einet = EinsumNetwork.EinsumNetwork(graph, args)

init_dict = None
if use_custom_initializer:
    init_dict = {}
    for layer in einet.einet_layers:
        init_dict[layer] = custom_initializer(layer, train_x)

einet.initialize(init_dict)
einet.to(device)
print(einet)


#################################
# Discriminative training phase #
#################################

optimizer = torch.optim.SGD(einet.parameters(), lr=SGD_learning_rate)
loss_function = torch.nn.CrossEntropyLoss()

train_N = train_x.shape[0]
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]

start_time = time.time()

for epoch_count in range(num_epochs):
    idx_batches = torch.randperm(train_N, device=device).split(batch_size)

    total_loss = 0
    for idx in idx_batches:
        batch_x = train_x[idx, :]
        optimizer.zero_grad()
        outputs = einet.forward(batch_x)
        target = torch.tensor([classes.index(train_labels[i]) for i in idx]).to(torch.device(device))
        loss = loss_function(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    print(f'[{epoch_count}]   total loss: {total_loss}')

end_time = time.time()

################
# Experiment 5 #
################

train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
print()
print("Experiment 5: Log-likelihoods  --- train LL {}   valid LL {}   test LL {}".format(
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))

################
# Experiment 6 #
################
train_labels = torch.tensor(train_labels).to(torch.device(device))
valid_labels = torch.tensor(valid_labels).to(torch.device(device))
test_labels = torch.tensor(test_labels).to(torch.device(device))

acc_train = EinsumNetwork.eval_accuracy_batched(einet, classes, train_x, train_labels, batch_size=batch_size)
acc_valid = EinsumNetwork.eval_accuracy_batched(einet, classes, valid_x, valid_labels, batch_size=batch_size)
acc_test = EinsumNetwork.eval_accuracy_batched(einet, classes, test_x, test_labels, batch_size=batch_size)
print()
print("Experiment 6: Classification accuracies  --- train acc {}   valid acc {}   test acc {}".format(
        acc_train,
        acc_valid,
        acc_test))

print()
print(f'Training time: {end_time - start_time}s')
