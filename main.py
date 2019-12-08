import csv

from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report
import seaborn as sn
from sklearn import metrics

from data_operations import ImagesGenerator, plot_cm
from models import NeuralNetworkModel


train = True


model = NeuralNetworkModel()
model.compile()


data_generator = ImagesGenerator()
train_generator, test_generator = data_generator.get_processed_images_train_generators()
test_labels = test_generator.labels


if train:
    mc = ModelCheckpoint('best_model_test.h5', monitor='val_accuracy', mode='max', save_best_only=True)
    history = model.model.fit_generator(train_generator, epochs=15, shuffle=True, verbose=1, validation_data=test_generator, callbacks=[mc])
else:
    model = load_model('best_model.h5')


# Get test generator without shuffle and test labels
test_generator = data_generator.get_processed_images_test_generator_and_labels()


predictions = model.model.predict_generator(test_generator)
predicted_labels = np.argmax(predictions, axis=-1)


# summarize history for accuracy and loss
if train:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['categorical_accuracy'], 'r')
    ax1.plot(history.history['val_categorical_accuracy'], 'b')
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

    print("Final training accuracy = {}".format(history.history["categorical_accuracy"][-1]))
    print("Final training loss = {}".format(history.history["loss"][-1]))


# accuracy, precision, recall and f1_score
accuracy = accuracy_score(test_labels, predicted_labels)
precision = precision_score(test_labels, predicted_labels, average="weighted", labels=np.unique(predicted_labels))
recall = recall_score(test_labels, predicted_labels, average="weighted")
fscore = f1_score(test_labels, predicted_labels, average="weighted", labels=np.unique(predicted_labels))


report = classification_report(test_labels, predicted_labels, labels=np.unique(predicted_labels))


with open('results.txt', 'w') as file:
    writer = csv.writer(file)
    writer.writerow([f"accuracy = {accuracy}"])
    writer.writerow([f"precision = {precision}"])
    writer.writerow([f"recall = {recall}"])
    writer.writerow([f"fscore = {fscore}"])
    writer.writerow([str(report)])


# displaying the confusion matrix using seaborn
array = metrics.confusion_matrix(test_labels, predicted_labels)
df_cm = pd.DataFrame(array, columns=np.unique(test_labels), index=np.unique(test_labels))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt="d") # font size
plt.show()


plot_cm(test_labels, predicted_labels)
plt.show()
