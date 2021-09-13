from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical, plot_model
from ResNetModel_layers import *
import numpy as np
import os

# network hyperparameters
batch_size = 200 
epochs = 50
num_classes = 10
resnet_blocks = 10

# load the cifar10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions
input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = resnet(input_shape=input_shape, resnet_blocks=resnet_blocks)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.summary()
plot_model(model, to_file='cifar10_ResNet.png', show_shapes=True)

# saving the model directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_ResNet_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# callbacks for model saving
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='acc', 
                             verbose=1, 
                             save_best_only=True)
callbacks = [checkpoint]

model.fit(x_train, y_train, 
          batch_size=batch_size, 
          verbose=1, 
          epochs=epochs, 
          validation_data=(x_test, y_test), 
          shuffle=True, 
          callbacks=callbacks)

scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print("\nTesting accuracy: %.1f%%" % (100.0 * scores[1]))
