import os
from os import listdir, makedirs
from os.path import join, exists, expanduser

# TODO usunac nieuzywane importy
from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report
import seaborn as sn
from sklearn import metrics

train = True
img_width, img_height = 112, 112

train_data_dir = './data/fruits_training/'
validation_data_dir = './data/fruits_test/'
nb_train_samples = 7225
nb_validation_samples = 2359
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_labels = test_generator.labels

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(112, 112, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

if train:
    mc = ModelCheckpoint('best_model2.h5', monitor='val_accuracy', mode='max', save_best_only=True)
    history = model.fit_generator(train_generator, epochs=15, shuffle=True, verbose=1, validation_data=test_generator, callbacks=[mc])
    #model.save('best_model.h5')
else:
    model = load_model('best_model2.h5')


test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)


predictions = model.predict_generator(test_generator)
predicted_labels = np.argmax(predictions, axis=-1)


# summarize history for accuracy and loss
if train:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['accuracy'], 'r')
    ax1.plot(history.history['val_accuracy'], 'b')
    ax1.set_title('Model accuracy')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    ax1.legend(['train', 'test'], loc='upper left')
    ax2.plot(history.history['loss'], 'r')
    ax2.plot(history.history['val_loss'], 'b')
    ax2.set_title('Model loss')
    ax2.set(xlabel='epoch', ylabel='loss')
    ax1.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig('his_acc_loss_mb2')

    print("Final training accuracy = {}".format(history.history["accuracy"][-1]))
    print("Final training loss = {}".format(history.history["loss"][-1]))


# accuracy, precision, recall and f1_score
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average="weighted", labels=np.unique(predicted_labels))
recall = recall_score(test_labels, predicted_labels, average="weighted")
fscore = f1_score(test_labels, predicted_labels, average="weighted", labels=np.unique(predicted_labels))

print(f"accuracy = {accuracy}")
print(f"precision = {precision}")
print(f"recall = {recall}")
print(f"fscore = {fscore}")

report = classification_report(test_labels, predicted_labels, labels=np.unique(predicted_labels))
print(report)


# displaying the confusion matrix using seaborn
array = metrics.confusion_matrix(test_labels, predicted_labels)
df_cm = pd.DataFrame(array, columns=np.unique(test_labels), index=np.unique(test_labels))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt="d") # font size
plt.show()

# seaborn heatmap more analysis
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
    sn.heatmap(cm, cmap="YlGnBu", annot=annot, annot_kws={"size": 10}, fmt='', ax=ax)

plot_cm(test_labels, predicted_labels)
plt.show()