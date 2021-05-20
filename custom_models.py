import keras
from keras import layers
from keras import regularizers
import tensorflow as tf
import numpy as np
from scipy import stats
import os
import math


def average_normalised_residual_error(W, Y, P, r, v, d):
    '''
    return the average normalised residual error over actual and estimated Y
    
    :param W: word embbeding matrix
    :type W: ndarray
    :param Y: meta embbeding matrix
    :type Y: ndarray
    :param P: transformation matrix
    :type P: ndarray
    :param r: number of models
    :type r: int
    :param v: vocab size
    :type v: int
    :param d:word embbeding dimension
    :type d: int
    
    :return: Average normalised residual error
    :rtype: float
    '''
    numerator = 0
    for w, p in zip(W, P):
        diff = Y - np.dot(w, p)
        norm = np.linalg.norm(diff)
        numerator += norm
        
    condition = (1.0 * numerator)/(r* math.sqrt(v*d))
        
    return condition


def ensemble_SOLS(W, Y, P, r, v, d, theta=0.001):
    '''
    return the meta embeeding matrix generated using SOLS (Solution with the Ordinary Least Squares) 
    
    :param W: word embbeding matrix
    :type W: ndarray
    :param Y: meta embbeding matrix
    :type Y: ndarray
    :param P: transformation matrix
    :type P: ndarray
    :param r: number of models
    :type r: int
    :param v: vocab size
    :type v: int
    :param d:word embbeding dimension
    :type d: int
    
    :return: meta embbeding matrix
    :rtype: ndarray    
    '''
    while average_normalised_residual_error(W[:], Y[:], P[:], r, v, d) > theta:
        Y = Y - np.mean(Y, axis=0)
        Y = Y/np.std(Y, axis=0)
        
        Y_new = np.zeros((v, d))
        for i in range(r):
            dot_inv = np.linalg.inv(np.dot(np.transpose(W[i]), W[i]))
            dot_1 = np.dot(dot_inv, np.transpose(W[i]))
            P[i] = np.dot(dot_1, Y)
            
            Y_new = np.add(Y_new, np.dot(W[i], P[i]))
            
        Y = Y_new/r
        
    return Y


def ensemble_SOPP(W, Y, P, r, v, d, theta=0.001):
    '''
    return the meta embeeding matrix generated using SOPP (Solution to the Orthogonal Procrustes Problem)
    
    :param W: word embbeding matrix
    :type W: ndarray
    :param Y: meta embbeding matrix
    :type Y: ndarray
    :param P: transformation matrix
    :type P: ndarray
    :param r: number of models
    :type r: int
    :param v: vocab size
    :type v: int
    :param d:word embbeding dimension
    :type d: int
    
    :return: meta embbeding matrix
    :rtype: ndarray
    '''
    while average_normalised_residual_error(W[:], Y[:], P[:], r, v, d) > theta:
        Y_new = np.zeros((v, d))
        for i in range(r):
            w_trans = np.transpose(W[i])
            S = np.dot(w_trans, Y)
            U, Ds, V_trans = np.linalg.svd(S, full_matrices=True, compute_uv=True)
            
            P[i] = np.dot(U, V_trans)
            
            Y_new = np.add(Y_new, np.dot(W[i], P[i]))
            
        Y = Y_new/r
        
    return Y


def model_1(d):
    '''
    build and return the autoencoder and encoder model
    
    :param d: word vector dimension
    :type d: int
    
    :return: tuple of autoencoder and encoder model
    :rtype:tuple of keras models
    '''
    def build_encoder_model(d, input_embb):
        '''
        build and return the encoder model
        
        :param d: word vector dimension
        :type d: int
        :param input_embb: input embbeding matrix
        :type input_embb: ndarray
        
        :return: encoder model
        :rtype: keras model
        '''
        encoded = layers.Dense(int(0.5*d), activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_embb)
        
        return encoded

    def build_decoder_model(encoded, d):
        '''
        build and return the decoder model
        
        :param d: word vector dimension
        :type d: int
        :param encoded: output of ecoder layer 
        :type encoded: ndarray
        
        :return: decoder model
        :rtype: keras model        
        '''
        decoded = layers.Dense(int(0.6*d), activation='relu')(encoded)
        decoded = layers.Dense(d, activation='relu')(decoded)
    
        return decoded

    
    input_embb_1 = keras.Input(shape=(d,), name='1')
    input_embb_2 = keras.Input(shape=(d,), name='2')
    input_embb_3 = keras.Input(shape=(d,), name='3')
    input_embb_4 = keras.Input(shape=(d,), name='4')
    input_embb_5 = keras.Input(shape=(d,), name='5')
    input_embb_6 = keras.Input(shape=(d,), name='6')
    input_embb_7 = keras.Input(shape=(d,), name='7')
    input_embb_8 = keras.Input(shape=(d,), name='8')
    input_embb_9 = keras.Input(shape=(d,), name='9')
    input_embb_10 = keras.Input(shape=(d,), name='10')
    
    encoded_1 = build_encoder_model(d, input_embb_1)
    encoded_2 = build_encoder_model(d, input_embb_2)
    encoded_3 = build_encoder_model(d, input_embb_3)
    encoded_4 = build_encoder_model(d, input_embb_4)
    encoded_5 = build_encoder_model(d, input_embb_5)
    encoded_6 = build_encoder_model(d, input_embb_6)
    encoded_7 = build_encoder_model(d, input_embb_7)
    encoded_8 = build_encoder_model(d, input_embb_8)
    encoded_9 = build_encoder_model(d, input_embb_9)
    encoded_10 = build_encoder_model(d, input_embb_10)
    
    shared_1 = layers.Concatenate()([encoded_1, encoded_2, encoded_3,encoded_4, encoded_5, 
                                     encoded_6,encoded_7, encoded_8, encoded_9, encoded_10])
    shared_2 = layers.Dense(d, activation='relu')(shared_1)
    shared_3 = layers.Dense(int(0.5*d), activation='relu')(shared_2)
    
    decoded_1 = build_decoder_model(shared_3, d)
    decoded_2 = build_decoder_model(shared_3, d)
    decoded_3 = build_decoder_model(shared_3, d)
    decoded_4 = build_decoder_model(shared_3, d)
    decoded_5 = build_decoder_model(shared_3, d)
    decoded_6 = build_decoder_model(shared_3, d)
    decoded_7 = build_decoder_model(shared_3, d)
    decoded_8 = build_decoder_model(shared_3, d)
    decoded_9 = build_decoder_model(shared_3, d)
    decoded_10 = build_decoder_model(shared_3, d)
    
    autoencoder = keras.Model([input_embb_1, input_embb_2, input_embb_3, input_embb_4, input_embb_5, 
                         input_embb_6, input_embb_7, input_embb_8, input_embb_9, input_embb_10],
                        [decoded_1, decoded_2, decoded_3, decoded_4, decoded_5, 
                         decoded_6, decoded_7, decoded_8, decoded_9, decoded_10])
    
    encoder = keras.Model([input_embb_1, input_embb_2, input_embb_3, input_embb_4, input_embb_5,
                           input_embb_6, input_embb_7, input_embb_8, input_embb_9, input_embb_10],
                          shared_2)
        
    return autoencoder, encoder


def model_2(d):
    '''
    build and return the autoencoder and encoder model
    
    :param d: word vector dimension
    :type d: int
    
    :return: tuple of autoencoder and encoder model
    :rtype:tuple of keras models
    '''    
    def build_encoder_model(d, input_embb):
        '''
        build and return the encoder model
        
        :param d: word vector dimension
        :type d: int
        :param input_embb: input embbeding matrix
        :type input_embb: ndarray
        
        :return: encoder model
        :rtype: keras model
        '''
        encoded = layers.Dense(int(0.6*d), activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_embb)
        encoded = layers.Dense(int(0.2*d), activation='relu', activity_regularizer=regularizers.l1(10e-5))(encoded)    

        return encoded

    def build_decoder_model(encoded, d):
        '''
        build and return the decoder model
        
        :param d: word vector dimension
        :type d: int
        :param encoded: output of ecoder layer 
        :type encoded: ndarray
        
        :return: decoder model
        :rtype: keras model        
        '''        
        decoded = layers.Dense(int(0.6*d), activation='relu')(encoded)
        decoded = layers.Dense(d, activation='relu')(decoded)
    
        return decoded

    
    input_embb_1 = keras.Input(shape=(d,), name='1')
    input_embb_2 = keras.Input(shape=(d,), name='2')
    input_embb_3 = keras.Input(shape=(d,), name='3')
    input_embb_4 = keras.Input(shape=(d,), name='4')
    input_embb_5 = keras.Input(shape=(d,), name='5')
    input_embb_6 = keras.Input(shape=(d,), name='6')
    input_embb_7 = keras.Input(shape=(d,), name='7')
    input_embb_8 = keras.Input(shape=(d,), name='8')
    input_embb_9 = keras.Input(shape=(d,), name='9')
    input_embb_10 = keras.Input(shape=(d,), name='10')
    
    encoded_1 = build_encoder_model(d, input_embb_1)
    encoded_2 = build_encoder_model(d, input_embb_2)
    encoded_3 = build_encoder_model(d, input_embb_3)
    encoded_4 = build_encoder_model(d, input_embb_4)
    encoded_5 = build_encoder_model(d, input_embb_5)
    encoded_6 = build_encoder_model(d, input_embb_6)
    encoded_7 = build_encoder_model(d, input_embb_7)
    encoded_8 = build_encoder_model(d, input_embb_8)
    encoded_9 = build_encoder_model(d, input_embb_9)
    encoded_10 = build_encoder_model(d, input_embb_10)
    
    shared_1 = layers.Concatenate()([encoded_1, encoded_2, encoded_3,encoded_4, encoded_5, 
                                     encoded_6,encoded_7, encoded_8, encoded_9, encoded_10])
    shared_2 = layers.Dense(d, activation='relu')(shared_1)
    shared_3 = layers.Dense(int(0.5*d), activation='relu')(shared_2)
    
    decoded_1 = build_decoder_model(shared_3, d)
    decoded_2 = build_decoder_model(shared_3, d)
    decoded_3 = build_decoder_model(shared_3, d)
    decoded_4 = build_decoder_model(shared_3, d)
    decoded_5 = build_decoder_model(shared_3, d)
    decoded_6 = build_decoder_model(shared_3, d)
    decoded_7 = build_decoder_model(shared_3, d)
    decoded_8 = build_decoder_model(shared_3, d)
    decoded_9 = build_decoder_model(shared_3, d)
    decoded_10 = build_decoder_model(shared_3, d)
    
    autoencoder = keras.Model([input_embb_1, input_embb_2, input_embb_3, input_embb_4, input_embb_5, 
                         input_embb_6, input_embb_7, input_embb_8, input_embb_9, input_embb_10],
                        [decoded_1, decoded_2, decoded_3, decoded_4, decoded_5, 
                         decoded_6, decoded_7, decoded_8, decoded_9, decoded_10])
    
    encoder = keras.Model([input_embb_1, input_embb_2, input_embb_3, input_embb_4, input_embb_5,
                           input_embb_6, input_embb_7, input_embb_8, input_embb_9, input_embb_10],
                          shared_2)
        
    return autoencoder, encoder


def model_3(d):
    '''
    build and return the autoencoder and encoder model
    
    :param d: word vector dimension
    :type d: int
    
    :return: tuple of autoencoder and encoder model
    :rtype:tuple of keras models
    '''    
    def build_encoder_model(d, input_embb):
        '''
        build and return the encoder model
        
        :param d: word vector dimension
        :type d: int
        :param input_embb: input embbeding matrix
        :type input_embb: ndarray
        
        :return: encoder model
        :rtype: keras model
        '''
        encoded = layers.Dense(int(0.8*d), activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_embb)
        encoded = layers.Dense(int(0.6*d), activation='relu', activity_regularizer=regularizers.l1(10e-5))(encoded)    
        encoded = layers.Dense(int(0.4*d), activation='relu', activity_regularizer=regularizers.l1(10e-5))(encoded)
    
        return encoded
    
    
    def build_decoder_model(encoded, d):
        '''
        build and return the decoder model
        
        :param d: word vector dimension
        :type d: int
        :param encoded: output of ecoder layer 
        :type encoded: ndarray
        
        :return: decoder model
        :rtype: keras model        
        '''
        decoded = layers.Dense(int(0.3*d), activation='relu')(encoded)
        decoded = layers.Dense(int(0.7*d), activation='relu')(decoded)
        decoded = layers.Dense(d, activation='relu')(decoded)
    
        return decoded

    
    input_embb_1 = keras.Input(shape=(d,), name='1')
    input_embb_2 = keras.Input(shape=(d,), name='2')
    input_embb_3 = keras.Input(shape=(d,), name='3')
    input_embb_4 = keras.Input(shape=(d,), name='4')
    input_embb_5 = keras.Input(shape=(d,), name='5')
    input_embb_6 = keras.Input(shape=(d,), name='6')
    input_embb_7 = keras.Input(shape=(d,), name='7')
    input_embb_8 = keras.Input(shape=(d,), name='8')
    input_embb_9 = keras.Input(shape=(d,), name='9')
    input_embb_10 = keras.Input(shape=(d,), name='10')
    
    encoded_1 = build_encoder_model(d, input_embb_1)
    encoded_2 = build_encoder_model(d, input_embb_2)
    encoded_3 = build_encoder_model(d, input_embb_3)
    encoded_4 = build_encoder_model(d, input_embb_4)
    encoded_5 = build_encoder_model(d, input_embb_5)
    encoded_6 = build_encoder_model(d, input_embb_6)
    encoded_7 = build_encoder_model(d, input_embb_7)
    encoded_8 = build_encoder_model(d, input_embb_8)
    encoded_9 = build_encoder_model(d, input_embb_9)
    encoded_10 = build_encoder_model(d, input_embb_10)
    
    shared_1 = layers.Concatenate()([encoded_1, encoded_2, encoded_3,encoded_4, encoded_5, 
                                     encoded_6,encoded_7, encoded_8, encoded_9, encoded_10])
    shared_2 = layers.Dense(2*d, activation='relu')(shared_1)
    shared_3 = layers.Dense(d, activation='relu')(shared_2)
    
    decoded_1 = build_decoder_model(shared_3, d)
    decoded_2 = build_decoder_model(shared_3, d)
    decoded_3 = build_decoder_model(shared_3, d)
    decoded_4 = build_decoder_model(shared_3, d)
    decoded_5 = build_decoder_model(shared_3, d)
    decoded_6 = build_decoder_model(shared_3, d)
    decoded_7 = build_decoder_model(shared_3, d)
    decoded_8 = build_decoder_model(shared_3, d)
    decoded_9 = build_decoder_model(shared_3, d)
    decoded_10 = build_decoder_model(shared_3, d)
    
    autoencoder = keras.Model([input_embb_1, input_embb_2, input_embb_3, input_embb_4, input_embb_5, 
                         input_embb_6, input_embb_7, input_embb_8, input_embb_9, input_embb_10],
                        [decoded_1, decoded_2, decoded_3, decoded_4, decoded_5, 
                         decoded_6, decoded_7, decoded_8, decoded_9, decoded_10])
        
    encoder = keras.Model([input_embb_1, input_embb_2, input_embb_3, input_embb_4, input_embb_5,
                           input_embb_6, input_embb_7, input_embb_8, input_embb_9, input_embb_10],
                          shared_3)
    
    return autoencoder, encoder


def model_4(d):
    '''
    build and return the autoencoder and encoder model
    
    :param d: word vector dimension
    :type d: int
    
    :return: tuple of autoencoder and encoder model
    :rtype:tuple of keras models
    '''
    def build_encoder_model(d, input_embb):
        '''
        build and return the encoder model
        
        :param d: word vector dimension
        :type d: int
        :param input_embb: input embbeding matrix
        :type input_embb: ndarray
        
        :return: encoder model
        :rtype: keras model
        '''
        encoded = layers.Dense(d, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_embb)
        
        return encoded

    def build_decoder_model(encoded, d):
        '''
        build and return the decoder model
        
        :param d: word vector dimension
        :type d: int
        :param encoded: output of ecoder layer 
        :type encoded: ndarray
        
        :return: decoder model
        :rtype: keras model        
        '''
#         decoded = layers.Dense(int(0.6*d), activation='relu')(encoded)
        decoded = layers.Dense(d, activation='relu')(encoded)
    
        return decoded

    
    input_embb_1 = keras.Input(shape=(d,), name='1')
    input_embb_2 = keras.Input(shape=(d,), name='2')
    input_embb_3 = keras.Input(shape=(d,), name='3')
    input_embb_4 = keras.Input(shape=(d,), name='4')
    input_embb_5 = keras.Input(shape=(d,), name='5')
    input_embb_6 = keras.Input(shape=(d,), name='6')
    input_embb_7 = keras.Input(shape=(d,), name='7')
    input_embb_8 = keras.Input(shape=(d,), name='8')
    input_embb_9 = keras.Input(shape=(d,), name='9')
    input_embb_10 = keras.Input(shape=(d,), name='10')
    
    encoded_1 = build_encoder_model(d, input_embb_1)
    encoded_2 = build_encoder_model(d, input_embb_2)
    encoded_3 = build_encoder_model(d, input_embb_3)
    encoded_4 = build_encoder_model(d, input_embb_4)
    encoded_5 = build_encoder_model(d, input_embb_5)
    encoded_6 = build_encoder_model(d, input_embb_6)
    encoded_7 = build_encoder_model(d, input_embb_7)
    encoded_8 = build_encoder_model(d, input_embb_8)
    encoded_9 = build_encoder_model(d, input_embb_9)
    encoded_10 = build_encoder_model(d, input_embb_10)
    
    shared_3 = layers.Average()([encoded_1, encoded_2, encoded_3,encoded_4, encoded_5, 
                                     encoded_6,encoded_7, encoded_8, encoded_9, encoded_10])
    
    decoded_1 = build_decoder_model(shared_3, d)
    decoded_2 = build_decoder_model(shared_3, d)
    decoded_3 = build_decoder_model(shared_3, d)
    decoded_4 = build_decoder_model(shared_3, d)
    decoded_5 = build_decoder_model(shared_3, d)
    decoded_6 = build_decoder_model(shared_3, d)
    decoded_7 = build_decoder_model(shared_3, d)
    decoded_8 = build_decoder_model(shared_3, d)
    decoded_9 = build_decoder_model(shared_3, d)
    decoded_10 = build_decoder_model(shared_3, d)
    
    autoencoder = keras.Model([input_embb_1, input_embb_2, input_embb_3, input_embb_4, input_embb_5, 
                         input_embb_6, input_embb_7, input_embb_8, input_embb_9, input_embb_10],
                        [decoded_1, decoded_2, decoded_3, decoded_4, decoded_5, 
                         decoded_6, decoded_7, decoded_8, decoded_9, decoded_10])
    
    encoder = keras.Model([input_embb_1, input_embb_2, input_embb_3, input_embb_4, input_embb_5,
                           input_embb_6, input_embb_7, input_embb_8, input_embb_9, input_embb_10],
                          shared_3)
        
    return autoencoder, encoder


def ensemble_CAEME_adv(W, autoencoder, encoder):
    '''
    train the autoencoder and return the trained autoencoder, encoder and the output of the encoder model
    
    :param W: list of word embbeding matrix
    :type W: ndarray
    :param autoencoder: built autoencoder model
    :type autoencoder: keras model
    :param encoder: built encoder model
    :type encoder: keras model
    
    :return: tuple of autoencoder model, encoder model and the output of the ecoder model
    :rtype: tuple
    '''
    d = W.shape[-1]
    r = W.shape[0]
    
#     autoencoder, encoder = build_model(d)
    
    optimizer = keras.optimizers.Adam(lr=0.001)
    autoencoder.compile(optimizer=optimizer, loss=['mse', 'mse', 'mse', 'mse', 'mse', 
                                          'mse', 'mse', 'mse', 'mse', 'mse'], 
                  loss_weights=[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    
    autoencoder.fit(list(W), list(W),
                epochs=100,
                batch_size=128,
                shuffle=True,
                   )
    
    encoded_embbs = encoder.predict(list(W))

    return autoencoder, encoder, encoded_embbs