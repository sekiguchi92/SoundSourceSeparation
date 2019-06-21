#!/usr/bin/env python3

import shutil
import os
import argparse
import pickle as pic
from progressbar import progressbar
import numpy as np

import chainer
from chainer import cuda, optimizers, serializers
from chainer.cuda import cupy as cp

import network_VAE
from configure_VAE import *

def train_VAE(gpu=GPU, dataset_fileName=DATASET_SAVE_PATH+'/wsj0_normalize_{}_{}.pic'.format(N_FFT, HOP_LENGTH)):
    file_suffix = "normal-scale=gamma-D={}".format(N_LATENT)

    if os.path.isfile(MODEL_SAVE_PATH + '/model-best-{0}.npz'.format(file_suffix) ):
        print(MODEL_SAVE_PATH + "model-best-{}.npz already exist".format(file_suffix))
        exit

    cuda.get_device_from_id(gpu).use()

    # Load dataset
    with open(dataset_fileName, 'rb') as f:
        dataset = pic.load(f)
    n_data = dataset.shape[1]

    # Prepare VAE model
    model = network_VAE.VAE(n_freq=int(N_FFT/2+1), n_latent=N_LATENT)
    model.to_gpu()

    # Setup Optimizer
    optimizer = optimizers.Adam(LEARNING_RATE)
    optimizer.setup(model)

    # Learning loop
    min_loss = np.inf
    loss_list = []
    for epoch in range(N_EPOCH):
        print('Epoch:', epoch+1)

        sum_loss = 0
        perm = np.random.permutation(n_data)
        for ii in progressbar(range(0, n_data, BATCH_SIZE)):
            minibatch = dataset[:, perm[ii:ii+BATCH_SIZE]].T
            scales = np.random.gamma(2, 0.5, (len(minibatch)))
            minibatch = minibatch * scales[:, None]
            x = chainer.Variable(cp.asarray(minibatch, dtype=cp.float32))

            optimizer.update(model.get_loss_func(), x)

            sum_loss += float(model.loss.data) * BATCH_SIZE
            loss_list.append(float(model.loss.data))

        sum_loss /= n_data
        print("Loss:", sum_loss)

        print('save the model and optimizer')
        serializers.save_npz(MODEL_SAVE_PATH + 'model-{0}.npz'.format(file_suffix), model)
        with open(MODEL_SAVE_PATH + 'loss-{0}.pic'.format(file_suffix), 'wb') as f:
            pic.dump(loss_list, f)

        if sum_loss < min_loss:
            shutil.copyfile(MODEL_SAVE_PATH + 'model-{0}.npz'.format(file_suffix), MODEL_SAVE_PATH + 'model-best-{0}.npz'.format(file_suffix))
            min_loss = sum_loss
        sum_loss = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=  int, default= GPU, help='GPU ID')
    args = parser.parse_args()

    from make_dataset_wsj0 import make_dataset
    make_dataset(WSJ0_PATH, DATASET_SAVE_PATH)

    train_VAE(args.gpu)
