import matplotlib.pyplot as plt
import numpy as np
import itertools
from cnn_model import ConvolutionalNN
from data_manipulation import ImageDataManipulation
from sklearn.metrics import confusion_matrix


def show_training_history(model, title="Model accuracy"):
    plt.plot(model.history['acc'])  # loss
    plt.plot(model.history['val_acc']) # val_loss
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def show_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def main():
    model = ConvolutionalNN()
    model.load_trained_model('model_cnn_4.json', 'model_cnn_4.h5')
    data_manip = ImageDataManipulation(None, None)
    test_data, test_labels = data_manip.load_training_data('test_2')
    dataset = data_manip.preprocess_data(test_data)
    result = model.predict_simple(dataset)
    y_pred = np.argmax(result, axis=1)
    test_labels = list(map(int, test_labels))
    cm = confusion_matrix(test_labels, y_pred)
    show_confusion_matrix(cm=cm,
                      normalize=True,
                      target_names=['anger', 'happy', 'sadness', 'surprise', 'neutral'],
                      title="Confusion Matrix")

#main()