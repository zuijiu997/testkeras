import tensorflow as tf
from tensorflow.contrib.keras import layers
import numpy as np


if __name__ == '__main__':


    input1_ = layers.Input(shape=(2, 2,3), name='input1')
    input2_ = layers.Input(shape=(2, 2,3), name='input2')

    y = layers.Concatenate()([input1_, input2_])
    model = tf.keras.Model(inputs=[input1_, input2_], outputs=y)
    model.summary()

    # 产生训练数据
    x1 = np.random.rand(1, 2, 2,3)
    print(x1)
    print('\n')
    x2 = np.random.rand(1, 2, 2,3)
    print(x2)
    print('\n')

    result = model.predict([x1, x2], batch_size=1)
    print(result)
    # print(result.reshape((6, 6)))
