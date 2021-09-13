from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input, Flatten, add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

#2D Convolution, Batch Normalization and Activation layers
def resnet_layers(inputs, num_filters=20, kernel_size=3, strides=1, activation='relu', batch_normalization=True):

    x = inputs
    x = Conv2D(filters = num_filters,
                  kernel_size = kernel_size,
                  strides = strides,
                  padding = 'same',
                  kernel_initializer = 'he_normal',
                  kernel_regularizer = l2(1e-4))(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x

def resnet(input_shape, resnet_blocks, num_classes=10):
    
    num_filters = 20

    inputs = Input(shape=input_shape)
    x = resnet_layers(inputs=inputs)

    for i in range(resnet_blocks):
        y = resnet_layers(inputs=x,
                            num_filters=num_filters)
        y = resnet_layers(inputs=y,
                            num_filters=num_filters,
                            activation=None)
        x = add([x, y])
        x = Activation('relu')(x)

    z = AveragePooling2D(pool_size=8)(x)
    z = Flatten()(z)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(z)

    model = Model(inputs=inputs, outputs=outputs)
    return model

