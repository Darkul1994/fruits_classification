import os

import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from sklearn import metrics

train_data_dir_default = './data/fruits_training/'
validation_data_dir_default = './data/fruits_test/'


class ImagesGenerator:
    def __init__(self, img_width=None, img_height=None, train_data_dir=None, validation_data_dir=None, batch_size=None):
        self.img_width = img_width if img_width else 112
        self.img_height = img_height if img_height else 112
        self.train_data_dir = train_data_dir if train_data_dir else train_data_dir_default
        self.validation_data_dir = validation_data_dir if validation_data_dir else validation_data_dir_default
        self.nb_train_samples = len([
            name for name in os.listdir(self.train_data_dir)
            if os.path.isfile(os.path.join(self.train_data_dir, name))
        ])
        self.nb_test_samples = len([
            name for name in os.listdir(self.validation_data_dir)
            if os.path.isfile(os.path.join(self.validation_data_dir, name))
        ])
        self.batch_size = batch_size if batch_size else 16

        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

    def get_processed_images_train_generators(self):
        train_generator = self.train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        test_generator = self.test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        return train_generator, test_generator

    def get_processed_images_test_generator_and_labels(self):
        test_generator = self.test_datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False
        )
        return test_generator


def plot_cm(test_labels, predicted_labels, figsize=(10, 10)):
    cm = metrics.confusion_matrix(test_labels, predicted_labels, labels=np.unique(test_labels))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(test_labels), columns=np.unique(test_labels))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    seaborn.heatmap(cm, cmap="YlGnBu", annot=annot, annot_kws={"size": 10}, fmt='', ax=ax)
