import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing


callback_loss = list()
callback_val_loss = list()


class LossCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        global callback_loss
        global callback_val_loss
        if epoch % 100 == 0:
            callback_loss = []
            callback_val_loss = []
        callback_loss.append(logs["loss"])
        callback_val_loss.append(logs["val_loss"])
        if epoch % 100 == 99:
            plt.figure()
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.plot(range(epoch-100, epoch), callback_loss, label='loss')
            plt.plot(range(epoch-100, epoch), callback_val_loss, label='val_loss')
            plt.legend()
            plt.savefig('/home/cillian/PycharmProjects/CBLassignment/Pics/pic_'+str(epoch)+'.png')



def plot_regression(x, y):
    plt.subplot(121)
    plt.scatter(test_list[0], test_list[1])
    plt.title('Test Data')
    plt.subplot(122)
    plt.plot(x, y, color='k')
    plt.title("Regression prediction")
    plt.show()


def plot_f(x, y, untrained):
    plt.subplot(121)
    plt.plot(x, untrained, color='r', label='untrained')
    plt.ylim([-10, 110])
    plt.legend(['untrained','Trained'])
    plt.plot(x, y, color='g', label='trained')
    plt.title('Untrained and trained prediction')
    plt.subplot(122)
    plt.scatter(test_list[0], test_list[1])
    plt.title('Test Data')
    plt.show()

def plot_loss_linear(r_history):
    plt.plot(r_history.history['loss'], label='loss')
    plt.plot(r_history.history['val_loss'], label='val_loss')
    plt.ylim([15, 30])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend('Linear Regression model')
    plt.grid(True)
    plt.show()


def plot_loss(r_history):
    plt.plot(r_history.history['loss'], label='loss')
    plt.plot(r_history.history['val_loss'], label='val_loss')
    plt.ylim([0, 30])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def text_to_list(a):
    txt_file = open(a, "r")
    to_list = []
    for line in txt_file:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        line_list = [float(x) for x in line_list]
        to_list.append(line_list)
    txt_file.close()
    return to_list


Epochs = 1000
train_file = "train.txt"
train_list = text_to_list(train_file)

test_file = "test.txt"
test_list = text_to_list(test_file)

plt.subplot(121)
plt.plot(train_list[0], train_list[1], "-o")
plt.title("Training data")
plt.subplot(122)
plt.plot(test_list[0], test_list[1], "-o")
plt.title("Testing data")
plt.show()

train_np = np.array(train_list[0])
normalizer = preprocessing.Normalization(input_shape=[1, ])
normalizer.adapt(train_np)


# Q1b: linear model:
def regression_model_linear(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation=None),
        layers.Dense(64, activation=None),
        layers.Dense(64, activation=None),
        layers.Dense(64, activation=None),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.SGD(0.01)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer)
    return model


linear_model = regression_model_linear(normalizer)
linear_model.summary()
predict_linear = linear_model.predict(test_list[0])
plot_regression(test_list[0], predict_linear)
checkpoint_path_linear = "/home/cillian/PycharmProjects/CBLassignment/Checkpoint_linear/cp.ckpt"
checkpoint_dir_linear = os.path.dirname(checkpoint_path_linear)
cp_callback_linear = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_linear,
                                                        save_weights_only=True,
                                                        verbose=0)
history = linear_model.fit(train_list[0],
                           train_list[1],
                           epochs=Epochs,
                           validation_data=(test_list[0], test_list[1]),
                           callbacks=[cp_callback_linear],
                           verbose=0)

os.listdir(checkpoint_dir_linear)
plot_loss_linear(history)
linear_model.load_weights(checkpoint_path_linear)
prediction = linear_model.predict(test_list[0])
plot_regression(test_list[0], prediction)
plot_f(test_list[0], prediction, predict_linear)


# Q2: regression model:
def a_relu(x):
    y = tf.math.maximum(x, 0)
    return y


def regression_model_relu(norm):
    model = keras.Sequential()
    model.add(norm)
    model.add(layers.Dense(64))
    model.add(layers.Activation(a_relu))
    model.add(layers.Dense(64))
    model.add(layers.Activation(a_relu))
    model.add(layers.Dense(64))
    model.add(layers.Activation(a_relu))
    model.add(layers.Dense(64))
    model.add(layers.Activation(a_relu))
    model.add(layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(0.01)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer)
    return model


# save weights:
callback_loss = []
r_model = regression_model_relu(normalizer)
r_model.summary()
predict_nonlinear = r_model.predict(test_list[0])
plot_regression(test_list[0], predict_nonlinear)
checkpoint_path = "ct"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)
# start training
history = r_model.fit(train_list[0],
                      train_list[1],
                      epochs=Epochs,
                      validation_data=(test_list[0], test_list[1]),
                      callbacks=[cp_callback, LossCallback()],
                      verbose=0)

os.listdir(checkpoint_dir)

plot_loss(history)
# Loads the weights
r_model.load_weights(checkpoint_path)
prediction = r_model.predict(test_list[0])
plot_regression(test_list[0], prediction)
# Re-evaluate the model
plot_f(test_list[0], prediction, predict_nonlinear)

x_relu = np.arange(-10, 10, 0.1)
y_relu = a_relu(x_relu)
plt.plot(x_relu, y_relu)
plt.title('Relu')
plt.show()

