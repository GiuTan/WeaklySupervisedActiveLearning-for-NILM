"""
@Author: Tamara Sobot
@Date: December 2023
"""

import argparse
import random as python_random
import numpy as np
import tensorflow as tf

from dummies import DummyModel, DummyDataGenerator
from AL_Loop import ALLoop
from utils import *


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--house", type=int, default=2)

    # Active learning arguments
    parser.add_argument("--acquisition-function", type=str, default='random')
    parser.add_argument("--queries-per-iteration", type=int, default=64)
    parser.add_argument("--max-queries-total", type=int, default=8192)
    parser.add_argument("--max-iterations", type=int, default=70)

    args = parser.parse_args()

    # Set path for saving indices of query pool, training and test samples.
    args.indices_path = 'AL_indices'

    # Create logging
    logs_dir = os.path.join('logs', get_filename(__file__))
    print('Logs in: {}'.format(logs_dir))
    logging = create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Set seeds for reproducible results
    np.random.seed(123)
    python_random.seed(123)
    tf.random.set_seed(1234)
    tf.experimental.numpy.random.seed(1234)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    model = DummyModel(args)  # modify for special model
    data_generator = DummyDataGenerator(args)  # modify for special data
    args.output_threshold = [0.5]  # modify for special data

    # Remove existing indices train/pool/test if there are any
    if os.path.exists(args.indices_path):
        to_remove = os.listdir(args.indices_path)
        for f in to_remove:
            os.remove(os.path.join(args.indices_path, f))

    # Remove the existing best model if there is any
    if os.path.exists('best_model.h5'):
        os.remove('best_model.h5')

    # Create AL loop instance.
    al_loop = ALLoop(args, model, data_generator)

    # Run AL
    for al_iteration in range(args.max_iterations):
        logging.info('*' * 100)

        if al_iteration == 0:
            al_loop.update_files(al_iteration)

            # Test the starting model
            t = datetime.now()
            metric_dict = al_loop.test()
            print(datetime.now() - t)
            logging.info('Initial AL iteration: {}'.format(al_iteration))
            logging.info('Initial metrics:\n {}'.format(metric_dict))

        # Make queries.
        al_loop.query(al_iteration=al_iteration + 1)

        # Update files containing indices of training and query pool samples.
        al_loop.update_files(al_iteration=al_iteration + 1)

        queried = np.load(os.path.join(args.indices_path, 'queried_indices_iter_{}.npy'.format(al_iteration + 1)))
        test = np.load(os.path.join(args.indices_path, 'test_indices.npy'))
        train = np.load(os.path.join(args.indices_path, 'training_indices.npy'))
        pool = np.load(os.path.join(args.indices_path, 'query_pool_indices.npy'))

        # If no samples are queried, exit the AL loop.
        if len(queried) == 0:
            print('No samples queried.')
            break

        # Log the number of samples.
        logging.info('AL iteration: {}'.format(al_iteration + 1))
        logging.info('Queried: {}'.format(len(queried)))
        logging.info('Total queried: {}'.format(len(train)))
        logging.info('Pool: {}'.format(len(pool)))

        # Train the model using new samples.
        al_loop.train()

        # Test the new best model and log results.
        metric_dict = al_loop.test()
        logging.info('Metrics: {}'.format(metric_dict))

    # Delete the best model.
    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')
