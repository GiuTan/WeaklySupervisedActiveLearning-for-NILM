"""
@Author: Tamara Sobot
@Date: December 2023
"""

import os.path
from AL_acquisition_functions import *


class ALLoop:
    def __init__(self, args, model, data_generator):
        self.args = args
        self.model = model
        self.data_generator = data_generator
        self.queried = None
        self.new_pool = None
        self.acquisition_function = args.acquisition_function

    def query(self, al_iteration):
        """
        Query samples that are most informative.
        Update data generator's training set and query pool using obtained indices.

        param al_iteration: number of the active learning iteration
        """
        print('=' * 50)
        print('Querying...')
        print('Acquisition function: {}'.format(self.acquisition_function))

        # Query - Get indices to be put into training set and updated list of indices for query pool.
        if self.acquisition_function == 'random':
            new_train_indices, new_pool_indices = random_query(self.data_generator, self.args)
        elif self.acquisition_function == 'pool_based_uncertainty':
            new_train_indices, new_pool_indices = pool_based_uncertainty(self.model, self.data_generator, self.args,
                                                                         al_iteration)
        elif self.acquisition_function == 'whole_pool':
            new_train_indices, new_pool_indices = whole_pool(self.data_generator)
        else:
            new_train_indices, new_pool_indices = [], []

        self.queried = new_train_indices
        self.new_pool = new_pool_indices

        if len(self.data_generator.train_indices):
            self.data_generator.train_indices = np.append(self.data_generator.train_indices, new_train_indices)
        else:
            self.data_generator.train_indices = new_train_indices

        self.data_generator.query_pool_indices = new_pool_indices

    def update_files(self, al_iteration):
        """
        Update .npy files containing training, pool and queried samples.
        For debugging/monitoring purposes.

        param queried: list of samples queried by the model.
        param al_iteration: number of the active learning iteration
        """

        if not os.path.exists(self.args.indices_path):
            os.makedirs(self.args.indices_path)

        # Save queried
        np.save(os.path.join(self.args.indices_path, 'queried_indices_iter_{}'.format(al_iteration)),
                self.queried)

        # Update training
        np.save(os.path.join(self.args.indices_path, 'training_indices'), self.data_generator.train_indices)

        # Update query pool
        np.save(os.path.join(self.args.indices_path, 'query_pool_indices'),
                self.data_generator.query_pool_indices)

        if not os.path.exists(os.path.join(self.args.indices_path, 'test_indices')):
            np.save(os.path.join(self.args.indices_path, 'test_indices'), self.data_generator.test_indices)

        print('Files containing indices updated.')

    def train(self):
        """
        Training procedure of the model.
        :return:
        """

        pass

    def test(self):
        """
        Testing procedure of the model.
        :return:
        """

        metrics = None

        return metrics
