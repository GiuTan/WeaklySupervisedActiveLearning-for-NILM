import numpy as np


class DummyModel:
    def __init__(self, args):
        self.args = args

    def train(self):
        pass

    def test(self):
        return None

    def predict(self, x):
        return 0


class DummyDataGenerator:
    def __init__(self, args):
        self.args = args
        self.x, self.y = self.load_data(args.data_path)

        # adjust for special case
        self.train_indices = np.arange(len(self.x)//4)
        self.query_pool_indices = np.arange(len(self.x)//4, len(self.x)//2)
        self.test_indices = np.array([])

    def load_data(self, data_path):
        x = np.ones((1000, 3, 100))
        y = np.ones((1000, 1))

        return x, y

