import tensorflow as tf
from tensorflow.contrib.keras import layers
import numpy as np

# from tensorflow.contrib import layers


if __name__ == '__main__':
    print(tf.VERSION)
    print(tf.keras.__version__)

    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.UpSampling2D(size=(2, 2)))

    data = np.random.random((1, 3, 3, 1))
    # print(data)
    print(data.reshape((3,3)))
    # labels = np.random.random((1000, 10))

    print('--------\n')
    print('--------\n')
    print('--------\n')
    result = model.predict(data, batch_size=1)
    print(result.shape)
    print(result.reshape((6,6)))
