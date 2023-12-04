from CRNN.params import params
from CRNN.CRNN_t import CRNN_construction
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from datetime import datetime
from CRNN.utils_func import thres_analysis, app_binarization_strong
from sklearn.metrics import classification_report


def get_model(args):
    lr = args.lr
    window_size = 2550
    weight = 1
    classes = 5

    model_ = 'solo_weakUK'
    drop = params[model_]['drop']
    kernel = params[model_]['kernel']
    num_layers = params[model_]['layers']
    gru_units = params[model_]['GRU']
    cs = params[model_]['cs']
    only_strong = params[model_]['no_weak']
    type_ = ''

    pre_trained = params[model_]['pre_trained']

    model = CRNN_construction(window_size, weight, lr=lr, classes=classes, drop_out=drop,
                              kernel=kernel, num_layers=num_layers, gru_units=gru_units, cs=cs,
                              path=pre_trained, only_strong=only_strong)

    return model, pre_trained


class CRNNTrainer:
    def __init__(self, args, model, data_generator):
        self.args = args
        self.model = model
        self.data_generator = data_generator

        self.thres_strong = None

    def train(self):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_strong_level_custom_f1_score', mode='max',
                                                      patience=20, restore_best_weights=True)
        log_dir_ = 'models/logs/logs_CRNN' + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=log_dir_)
        file_writer = tf.summary.create_file_writer(log_dir_ + "/metrics")
        file_writer.set_as_default()

        self.model.load_weights(self.args.pretrained_model_path)

        history = self.model.fit(x=self.data_generator.aggregate[self.data_generator.train_indices],
                                 y=[np.negative(
                                     np.ones(self.data_generator.labels[self.data_generator.train_indices].shape)),
                                     # Training with weak labels only
                                     self.data_generator.weak_labels[self.data_generator.train_indices]],
                                 shuffle=True,
                                 epochs=1000,
                                 batch_size=self.args.batch_size,
                                 validation_data=(self.data_generator.aggregate[self.data_generator.test_indices],
                                                  [self.data_generator.labels[self.data_generator.test_indices],
                                                   self.data_generator.weak_labels[self.data_generator.test_indices]]),
                                 callbacks=[early_stop, tensorboard],
                                 verbose=1)
        self.model.save_weights('best_model.h5')
        print('Model saved...')

    def test(self):
        # Get model outputs.
        output_strong, output_weak = self.model.predict(
            x=self.data_generator.aggregate[self.data_generator.test_indices])
        label_strong = self.data_generator.labels[self.data_generator.test_indices]

        # Reshape.
        shape = output_strong.shape[0] * output_strong.shape[1]
        output_strong = output_strong.reshape(shape, 5)
        label_strong = label_strong.reshape(shape, 5)

        assert (label_strong.shape == output_strong.shape)

        self.thres_strong = thres_analysis(label_strong, output_strong, 5)

        output_strong = app_binarization_strong(output_strong, self.thres_strong, 5)

        if self.args.house == 2:
            label_strong = np.delete(label_strong, [1, 2], axis=-1)
            output_strong = np.delete(output_strong, [1, 2], axis=-1)
        elif self.args.house == 4:
            label_strong = np.delete(label_strong, [2, 3, 4], axis=-1)
            output_strong = np.delete(output_strong, [2, 3, 4], axis=-1)
        elif self.args.house == 5:
            label_strong = np.delete(label_strong, [1, 2], axis=-1)
            output_strong = np.delete(output_strong, [1, 2], axis=-1)
        elif self.args.house == 19:
            label_strong = np.delete(label_strong, [2, 3, 4], axis=-1)
            output_strong = np.delete(output_strong, [2, 3, 4], axis=-1)

        print(np.sum(label_strong, axis=0))

        return classification_report(label_strong, output_strong)
