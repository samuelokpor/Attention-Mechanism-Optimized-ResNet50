import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Input
from keras.layers import Activation, Concatenate, Conv2D, Multiply

def channel_attention_module(x, ratio=8):
    b, _,_,channel = x.shape
    #shared layers
    l1 = Dense(channel//ratio, activation="relu", use_bias=False)
    l2 = Dense(channel, use_bias=False)

    ##GLobal average Pooling
    x1 = GlobalAveragePooling2D()(x)
    x1 = l1(x1)
    x1 = l2(x1)
    
    #global Max Pooling
    x2 = GlobalMaxPooling2D()(x)
    x2 = l1(x2)
    x2 = l2(x2)

    ##Add both and apply sigmoid
    features = x1 + x2
    features = Activation("sigmoid")(features)
    features = Multiply()([x, features])

    return features


def spatial_attention_module(x):
    ## average Pooling
    x1 = tf.reduce_mean(x, axis=-1)
    x1 = tf.expand_dims(x1, axis=-1)

    #max Pooling
    x2 = tf.reduce_max(x, axis=-1)
    x2 = tf.expand_dims(x2, axis=-1)

    ## Concatenate
    features = Concatenate()([x1,x2])

    ##Conv Layer
    features = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(features)
    features = Multiply()([x, features])

    return features

def cbam(x):
    x = channel_attention_module(x)
    x = spatial_attention_module(x)
    return x

inputs = Input(shape=(224,224,3))
y = cbam(inputs)
print(y.shape)
