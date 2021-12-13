import tensorflow as tf
import tensorflow.keras.losses as losses
from itertools import cycle
import numpy as np
import pandas as pd
import model as md
import data_processing
import os
import itertools
from sacred import Experiment
from sacred.observers import FileStorageObserver
import matplotlib.pyplot as plt
from shutil import copy
import json


# logging of the experiments is done with sacred


def train_experiment(model, train_ds, val_ds, epochs):
    model.summary()
    print("Starting model training...")
    # start training:
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return history


def save_losses(history, save_path):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(save_path)
    plt.clf()


def create_datasets(data_dir, img_size, batch_size, channel_idx=1):
    ch_str = ["back", "front", "top"]
    channel = ch_str[channel_idx]
    train_data, val_data, test_data = data_processing.create_data_sets(data_dir, channel, batch_size, image_size=img_size)
    num_classes = len(train_data.class_names)
    assert len(train_data.class_names) == len(val_data.class_names)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.AUTOTUNE)
    return train_data, val_data, test_data, num_classes


def run_layer_filter_experiments(layers, filters, image_size, batch_size, kernels, data_dir=None, out_dir=os.getcwd(),
                                 epochs=3, deep=False, pool=False):
    if data_dir is None:
        data_dir = os.getcwd() + r'\data'
    run_path = os.path.join(out_dir)

    ex = Experiment('Varying layers and filters')
    ex.observers.append(FileStorageObserver(basedir=os.path.join(run_path, ex.path)))
    # loss_path = os.path.join(run_path, ex.path, 'losses.png')

    @ex.config
    def config():
        """This is the configuration of the experiment"""
        dataset_name = 'Anomaly Detection'
        net_architecture = 'Simple CNN'
        data_dir = os.getcwd() + r'\data'
        train_ds = []
        val_ds = []
        n_layers = []
        n_filters = []
        image_size = []
        batch_size = []
        optimizer = 'adam'
        epochs = []
        kernel = []

    @ex.capture
    def build_model(n_layers, filters, image_size, kernel, num_classes):
        model = md.build_conv_network(n_layers, filters, kernel=kernel, image_size=image_size, classes=num_classes)
        return model

    @ex.capture
    def train(model, train_ds, val_ds, epochs):
        # create callback to save best model:
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ex.observers[0].dir, 'saved_model/best/best_model.h5'),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # add early stopping criterion:
        early_stop = tf.keras.callbacks.EarlyStopping(patience=8)
        # start training:
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[save_best, early_stop])
        model.save(os.path.join(ex.observers[0].dir, 'saved_model/last_model.h5'))

        return history

    @ex.capture
    def save_losses(history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig(os.path.join(ex.observers[0].dir, 'losses.png'))
        plt.clf()

    @ex.capture
    def test(model, test_data):
        result = model.evaluate(test_data)
        return result

    @ex.main
    def main():
        # Get data
        print('Creating data-sets...')
        train_ds, val_ds, test_ds, num_classes = create_datasets(data_dir, image_size, batch_size)

        # build network
        print('Building model...')
        model = build_model(num_classes=num_classes)

        # train network
        print('Training network:')
        history = train(model, train_ds, val_ds)

        # Save plot with losses
        print('Saving losses...')
        save_losses(history)
        print('Final accuracy: ', history.history['val_accuracy'][-1])

        # run test
        result = model.evaluate(test_ds)

    results = {}
    print('layers: {}, filters: {}'.format(layers, filters))
    conf = {'n_layers': int(layers),
            'filters': filters,
            'image_size': image_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'kernel': kernels}

    exp_finish = ex.run(config_updates=conf)
    results['layers: {}, filters: {}'.format(layers, filters)] = exp_finish.result


if __name__ == "__main__":
    # CNN parameters
    layers =    [10]
    filters =   [16]
    kernel =    3

    epochs = 30
    batch_size = 8
    image_size = (360, 640)
    data_dir = r'C:\Users\marni\Documents\Pakt\data'
    out_dir = r'C:\Users\marni\Documents\Pakt\results'

    for layer, filter in zip(layers, filters):
        run_layer_filter_experiments(layer, filter, kernels=kernel, image_size=image_size, batch_size=batch_size,
                                     data_dir=data_dir, epochs=epochs, out_dir=out_dir)
