import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork, FactorizedLeafLayer
from EinsumNetwork.initializations import get_init_dict
from experiments.round1 import Settings
import datasets
import utils
import time
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

demo_text = """
This demo loads (fashion) mnist and quickly trains an EiNet for some epochs. 

There are some parameters to play with, as for example which exponential family you want 
to use, which classes you want to pick, and structural parameters. Then an EiNet is trained, 
the log-likelihoods reported, some (conditional and unconditional) samples are produced, and
approximate MPE reconstructions are generated. 

Experiment 1:
A single Einet is learned by optimizing the log likelihood by Expectation Maximization.
The log likelihood is measured.

Experiment 2:
A single Einet is learned by optimizing the log likelihood by Expectation Maximization. 
A classification task is executed.
"""
print(demo_text)

def run_experiment(settings):
    ############################################################################

    fashion_mnist = settings.fashion_mnist
    svhn = settings.svhn

    exponential_family = settings.exponential_family

    classes = settings.classes

    K = settings.K

    structure =  settings.structure

    # 'poon-domingos'
    pd_num_pieces = settings.pd_num_pieces

    # 'binary-trees'
    depth = settings.depth
    num_repetitions = settings.num_repetitions

    width = settings.width
    height = settings.height

    num_epochs = settings.num_epochs
    batch_size = settings.batch_size
    online_em_frequency = settings.online_em_frequency
    online_em_stepsize = settings.online_em_stepsize

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
    elif svhn:
        train_x, train_labels, test_x, test_labels, extra_x, extra_labels = datasets.load_svhn()
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

    # Make EinsumNetwork
    ######################################
    if structure == 'poon-domingos':
        pd_delta = [[height / d, width / d] for d in pd_num_pieces]
        graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
    elif structure == 'binary-trees':
        graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
    else:
        raise AssertionError("Unknown Structure")

    args = EinsumNetwork.Args(
            num_var=train_x.shape[1],
            num_dims=3 if svhn else 1,
            num_classes=1,
            num_sums=K,
            num_input_distributions=K,
            exponential_family=exponential_family,
            exponential_family_args=exponential_family_args,
            online_em_frequency=online_em_frequency,
            online_em_stepsize=online_em_stepsize)

    einet = EinsumNetwork.EinsumNetwork(graph, args)

    init_dict = get_init_dict(einet, train_x)
    einet.initialize(init_dict)
    einet.to(device)
    print(einet)

    num_params = EinsumNetwork.eval_size(einet)

    # Train
    ######################################

    train_N = train_x.shape[0]
    valid_N = valid_x.shape[0]
    test_N = test_x.shape[0]

    start_time = time.time()

    for epoch_count in range(num_epochs):
        idx_batches = torch.randperm(train_N, device=device).split(batch_size)

        total_ll = 0.0
        for idx in idx_batches:
            batch_x = train_x[idx, :]
            outputs = einet.forward(batch_x)
            ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood = ll_sample.sum()
            log_likelihood.backward()

            einet.em_process_batch()
            total_ll += log_likelihood.detach().item()
        einet.em_update()
        print(f'[{epoch_count}]   total log-likelihood: {total_ll/train_N}')

    end_time = time.time()

    ################
    # Experiment 1 #
    ################
    einet.eval()
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
    print()
    print("Experiment 1: Log-likelihoods  --- train LL {}   valid LL {}   test LL {}".format(
            train_ll / train_N,
            valid_ll / valid_N,
            test_ll / test_N))

    ################
    # Experiment 2 #
    ################
    train_labels = torch.tensor(train_labels).to(torch.device(device))
    valid_labels = torch.tensor(valid_labels).to(torch.device(device))
    test_labels = torch.tensor(test_labels).to(torch.device(device))

    acc_train = EinsumNetwork.eval_accuracy_batched(einet, classes, train_x, train_labels, batch_size=batch_size)
    acc_valid = EinsumNetwork.eval_accuracy_batched(einet, classes, valid_x, valid_labels, batch_size=batch_size)
    acc_test = EinsumNetwork.eval_accuracy_batched(einet, classes, test_x, test_labels, batch_size=batch_size)
    print()
    print("Experiment 2: Classification accuracies  --- train acc {}   valid acc {}   test acc {}".format(
            acc_train,
            acc_valid,
            acc_test))

    print()
    print(f'Network size: {num_params} parameters')
    print(f'Training time: {end_time - start_time}s')

    return {
        'train_ll': train_ll / train_N,
        'valid_ll': valid_ll / valid_N,
        'test_ll': test_ll / test_N,
        'train_acc': acc_train,
        'valid_acc': acc_valid,
        'test_acc': acc_test,
        'network_size': num_params,
        'training_time': end_time - start_time,
    }



start_total = time.time()


settings = Settings.Settings()
if settings.save_data:
    dataset_list = []
    for dataset in settings.datasets:
        settings.set_dataset(dataset)
        size_list = []
        for size in settings.sizes:
            settings.set_size(size)
            print(f"""
            Running experiment for network size: {size} and dataset: {dataset}
            """)
            s = {}
            s[size] = run_experiment(settings)
            size_list.append(s)
        d = {}
        d[dataset] = size_list
        dataset_list.append(d)
    
    out = {}
    out['experiment_1_2'] = dataset_list

    # save in file
    data_dir = '../src/experiments/round1/data'
    data_dir = '../src/experiments/round1/data/all-classes'
    utils.mkdir_p(data_dir)
    data_file = os.path.join(data_dir, "experiment_1_2.json")
    with open(data_file, 'w') as f:
        json.dump(out, f)       
else: 
    run_experiment(settings)


end_total = time.time()
print()
print(f'Total running time: {end_total - start_total}s')
