import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.callbacks import TensorBoard
import tensorflow as tf

device_name = tf.test.gpu_device_name()

root_dir = 'assignment-2-video-search\\3-embedding-model\model-input-dataset'

# ---------------------------------------------------------
# ---------------------------------------------------------
# def ConvAutoencoder():
#     input_img = keras.Input(shape=(28, 28, 1))

#     x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
#     x = layers.MaxPooling2D((2, 2), padding='same')(x)
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D((2, 2), padding='same')(x)
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#     encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

#     # at this point the representation is (4, 4, 8) i.e. 128-dimensional

#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#     x = layers.UpSampling2D((2, 2))(x)
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#     x = layers.UpSampling2D((2, 2))(x)
#     x = layers.Conv2D(16, (3, 3), activation='relu')(x)
#     x = layers.UpSampling2D((2, 2))(x)
#     decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

#     autoencoder = keras.Model(input_img, decoded)
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     return autoencoder

def down(filters , kernel_size, apply_batch_normalization=True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters, kernel_size, padding='same', strides=2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(layers.LeakyReLU())
    return downsample

def up(filters, kernel_size, apply_dropout=False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size, padding='same', strides=2))
    if apply_dropout:
        upsample.add(layers.Dropout(0.2))
    upsample.add(layers.LeakyReLU())
    return upsample

def ConvAutoencoder():
    inputs = layers.Input(shape=[128, 128, 1])
    d1 = down(128, (3, 3), False)(inputs)
    d2 = down(128, (3, 3), False)(d1)
    d3 = down(256, (3, 3), True)(d2)
    
    # Upsampling
    u1 = up(256, (3, 3), False)(d3)
    u1 = layers.concatenate([u1, d2])
    u2 = up(128, (3, 3), False)(u1)
    u2 = layers.concatenate([u2, d1])
    u3 = up(1, (3, 3), False)(u2)
    u3 = layers.concatenate([u3, inputs])
    
    output = layers.Conv2D(1, (2, 2), strides=1, padding='same')(u3)
    autoencoder = tf.keras.Model(inputs=inputs, outputs=output)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder

# ---------------------------------------------------------
# ---------------------------------------------------------

img_arr = "assignment-2-video-search\\3-embedding-model\mnistlikedataset128x1.npz"
with np.load(img_arr) as data:
    #load DataX as train_data
    data = data['DataX']
    np.random.shuffle(data)
    portion = int(0.8*len(data))
    x_train, x_test = data[:portion,:], data[portion:,:]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))
    x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))
    print(x_train.shape)
    print(x_test.shape)

# ---------------------------------------------------------
# ---------------------------------------------------------

model = ConvAutoencoder()
model.fit(x_train, x_train,
          epochs=100,
          batch_size=128,
          shuffle=True,
          validation_data=(x_test, x_test),
          callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
# Use tensorboard --logdir=/tmp/autoencoder in cli to view graph
model.save("autoencoder_model.keras")

# ---------------------------------------------------------
# ---------------------------------------------------------

# model_path = 'assignment-2-video-search\\3-embedding-model\conv_autoencoder.keras'
# model = keras.models.load_model(model_path)

# decoded_imgs = model.predict(x_test)