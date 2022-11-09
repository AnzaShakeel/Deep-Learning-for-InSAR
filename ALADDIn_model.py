 
import keras
import keras.backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling3D, UpSampling3D, Activation, Dense, Reshape, Flatten,Concatenate
from keras.layers import Conv2DTranspose, ConvLSTM2D, TimeDistributed, Lambda, Bidirectional, Add
from keras_layer_normalization import LayerNormalization
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model
from lambda_layers import create_IFG
import warnings
warnings.filterwarnings('ignore')


def create_model(input_img):
  
    #### Gen 1
    #inp = Reshape(target_shape=(256,256,26))(input_img)
    x1 = TimeDistributed(Conv2D(32, (1, 1), activation="tanh",padding="same", kernel_initializer='glorot_normal'), batch_input_shape=(None, 26, 256, 256, 1))(input_img)
    x1 = LayerNormalization()(x1)
    x1 = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x1)
    
    x2 = TimeDistributed(Conv2D(64, (3, 3), activation="tanh",padding="same", kernel_initializer='glorot_normal'))(x1)
    x2 = LayerNormalization()(x2)
    c1 = Concatenate(axis=-1)([x2,x1])
    x3 = TimeDistributed(Conv2D(64, (3, 3), activation="tanh",padding="same", kernel_initializer='glorot_normal'))(c1)
    x3 = LayerNormalization()(x3)
    x3 = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x3)
    
    x4 = TimeDistributed(Conv2D(128, (3, 3), activation="tanh",padding="same", kernel_initializer='glorot_normal'))(x3)
    x4 = LayerNormalization()(x4)
    c2 = Concatenate(axis=-1)([x4, x3])
    x5 = TimeDistributed(Conv2D(128, (3, 3), activation="tanh",padding="same", kernel_initializer='glorot_normal'))(c2)
    x5 = LayerNormalization()(x5)
    x5 = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x5)

    # # # # #
    x6 = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh')(x5)
    x6 = LayerNormalization()(x6)
    c3 = Concatenate(axis=-1)([x6, x5])
    x7 = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh')(c3)
    x7 = LayerNormalization()(x7)
    x7 = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x7)
    x8 = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh')(x7)
    x8 = LayerNormalization()(x8)
    c4 = Concatenate(axis=-1)([x8, x7])
    x9 = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh')(c4)
    x9 = LayerNormalization()(x9)
    x9 = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x9)

    # # # # #
    #shape = K.int_shape(x1)
    x = Flatten()(x9)
    x = Dense(2048, kernel_initializer='glorot_normal',activation="linear")(x)
    x = Dense(576, kernel_initializer='glorot_normal',activation="linear")(x)
 
    x10 = Reshape((9,8,8,1))(x)
    
    x11 = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh')(x10)
    x11 = LayerNormalization()(x11)
    c5 = Concatenate(axis=-1)([x11, x10])
    x12 = ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh')(c5)
    x12 = LayerNormalization()(x12)
    x12 = UpSampling3D(size=(1, 2, 2))(x12)
    x13 = ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh')(x12)
    x13 = LayerNormalization()(x13)
    c6 = Concatenate(axis=-1)([x13, x12])
    x14 =  ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh')(c6)
    x14 = LayerNormalization()(x14)
    x14 = UpSampling3D(size=(1, 2, 2))(x14)
   
    x15 = TimeDistributed(Conv2DTranspose(128, (3, 3), activation="tanh",padding="same", kernel_initializer='glorot_normal'), batch_input_shape=(None, 9, 256, 256, 1))(x14)
    x15 = LayerNormalization()(x15)
    c7 = Concatenate(axis=-1)([x15, x14])
    x16 = TimeDistributed(Conv2DTranspose(128, (3, 3), strides =2, activation="tanh",padding="same", kernel_initializer='glorot_normal'))(c7)
    x16 = LayerNormalization()(x16)
    x17 = TimeDistributed(Conv2DTranspose(64, (3, 3), activation="tanh", padding="same", kernel_initializer='glorot_normal'))(x16)
    x17 = LayerNormalization()(x17)
    c8 = Concatenate(axis=-1)([x17, x16])
    x18 = TimeDistributed(Conv2DTranspose(64, (3, 3), strides =2, activation="tanh", padding="same", kernel_initializer='glorot_normal'))(c8)
    x18 = LayerNormalization()(x18)
    x19 = TimeDistributed(Conv2DTranspose(32, (3, 3), strides =2, activation="tanh", padding="same", kernel_initializer='glorot_normal'))(x18)
    x19 = LayerNormalization()(x19)
    
    
    x20 =  Bidirectional(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, kernel_initializer='glorot_normal',recurrent_activation='hard_sigmoid', activation='tanh'))(x19)
    x20 = LayerNormalization()(x20)
    TS = TimeDistributed(Conv2D(1, (1, 1), activation="linear", padding="same", kernel_initializer='glorot_normal'), name = 'TS')(x20)
    
    IFG = Lambda(create_IFG, name ='create_IFG')(TS)
    
    return IFG
