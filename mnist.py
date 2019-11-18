import numpy as np
import matplotlib.pyplot as plt
# from keras.utils import np_utils

from DgtR import DigitRecignizer


def load_data(path):
    with np.load(path) as file:
        x_train, y_train = file['x_train'], file['y_train']
        x_test, y_test = file['x_test'], file['y_test']
        return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data('../datasets/mnist.npz')

x_train = x_train / 255
x_test = x_test / 255

model = DigitRecignizer()

model.fit(x_train, y_train, 0.1)
print(model.model)
pred = model.predict(x_test)




# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# index = 12
# plt.imshow(x_train[index, :, :])
# plt.show()
#
# print(y_train[index])
#
# Y_train = np_utils.to_categorical(y_train, 10)
# Y_test = np_utils.to_categorical(y_test, 10)
#
# model = DigitRecignizer(3)
# model.fit(x_train, y_train)
# pred = model.predict(x_test)
#
accuracy = np.sum(pred == y_test) / len(y_test)
print(accuracy * 100)
