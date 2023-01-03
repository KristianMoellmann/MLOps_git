# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import torch
import pickle
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    train = {'images': [], 'labels': []}
    for i in range(5):
        imgs = np.load(f"{input_filepath}/train_{i}.npz")['images']
        labels = np.load(f"{input_filepath}/train_{i}.npz")['labels']
        train['images'].append(imgs)
        train['labels'].append(labels)
    train['images'] = np.concatenate(train['images'])
    train['labels'] = np.concatenate(train['labels'])
    train_mu, train_std = np.mean(train['images'], axis=0), np.std(train['images'], axis=0)
    train['images'] = (train['images'] - train_mu) / (train_std + 1e-8)
    train['images'] = torch.from_numpy(train['images'])
    train['labels'] = torch.from_numpy(train['labels'])

    with open(f'{output_filepath}/train.pickle', 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    test = {'images': [], 'labels': []}
    test['images'] = np.load(f"{input_filepath}/test.npz")['images']
    test['labels'] = np.load(f"{input_filepath}/test.npz")['labels']
    test['images'] = (test['images'] - train_mu) / (train_std + 1e-8)
    test['images'] = torch.from_numpy(test['images'])
    test['labels'] = torch.from_numpy(test['labels'])
    with open(f'{output_filepath}/test.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
