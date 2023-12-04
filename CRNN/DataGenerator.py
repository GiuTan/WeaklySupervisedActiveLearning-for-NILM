import os
import numpy as np
from natsort import natsorted
from CRNN.params import uk_params


class DataGenerator(object):
    def __init__(self, args):
        self.args = args

        # Data normalization parameters
        self.train_mean = uk_params['mean']
        self.train_std = uk_params['std']

        # Load data and create weak labels
        self.aggregate, self.labels = self.load_data()
        self.weak_labels = self.compute_weak_labels()

        # Set weak labels to -1 for appliances that are not present in the house.

        if self.args.house == 2:
            self.weak_labels[:, :, 1:3] = -1
            self.labels[:, :, 1:3] = -1
        elif self.args.house == 4:
            self.weak_labels[:, :, 2:] = -1
            self.labels[:, :, 2:] = -1
        elif self.args.house == 5:
            self.weak_labels[:, :, 1:3] = -1
            self.labels[:, :, 1:3] = -1
        elif self.args.house == 19:
            self.weak_labels[:, :, 2:] = -1
            self.labels[:, :, 2:] = -1

        print('Agg, strong and weak label shape: ', self.aggregate.shape, self.labels.shape, self.weak_labels.shape)

        # Normalize data
        self.aggregate = (self.aggregate - self.train_mean) / self.train_std

        # Set training, query pool and testing indices
        self.train_indices = []
        self.query_pool_indices = np.arange(0, int(0.3 * len(self.aggregate)))
        self.test_indices = np.arange(int(0.3 * len(self.aggregate)), len(self.aggregate))

    def load_data(self):
        """
        Load data (aggregate + strong labels) from resampled_REFIT_test.
        """
        # Get files
        agg_files = natsorted(
            os.listdir(os.path.join(self.args.data_path, 'agg/house_{}'.format(self.args.house))))
        label_files = natsorted(
            os.listdir(os.path.join(self.args.data_path, 'labels/house_{}'.format(self.args.house))))

        # Read files and stack them
        aggregate = np.stack(
            [np.load(
                os.path.join(self.args.data_path, 'agg/house_{}'.format(self.args.house), f))
                for f in agg_files if
                np.load(
                    os.path.join(self.args.data_path, 'agg/house_{}'.format(self.args.house),
                                 f)).shape == (2550,)
            ],
            axis=0)

        labels = np.stack(
            [np.load(
                os.path.join(self.args.data_path, 'labels/house_{}'.format(self.args.house), f),
                allow_pickle=True)
                for f in label_files if
                np.load(
                    os.path.join(self.args.data_path, 'labels/house_{}'.format(self.args.house),
                                 f),
                    allow_pickle=True).shape == (5, 2550)
            ],
            axis=0)

        return aggregate, labels.transpose(0, 2, 1)

    def compute_weak_labels(self):
        """
        Compute weak labels according to linear max pooling layer function.
        """
        print(self.labels.shape)

        square = np.square(self.labels)
        sum_square = np.sum(square, axis=1, keepdims=True)

        sum_ = np.sum(self.labels, axis=1, keepdims=True)

        return np.divide(sum_square, sum_, out=np.zeros(sum_.shape), where=sum_ != 0)
