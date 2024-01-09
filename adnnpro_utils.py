from keras.layers import concatenate, Flatten
from keras import optimizers
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Input, Add, multiply, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D

from keras import backend as K

import os

import numpy as np

from sklearn.preprocessing import LabelBinarizer
import keras.preprocessing.sequence as kps

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _get_onehot(seqs_list):
    res = [x.upper() for x in seqs_list]
    seqs_list = res
    lb = LabelBinarizer()
    pro = ['A', 'C', 'G', 'T']
    lb.fit(pro)
    one_hot_code = []
    for value in seqs_list:
        seq = lb.transform(list(value))
        one_hot_code.append(seq)
    encode = kps.pad_sequences(one_hot_code, maxlen=1000, dtype='int32', padding='post', truncating='post')

    return encode


def _read(input_file):
    with open(input_file, 'r') as f:
        fasta = {}
        for line in f:
            line = line.strip()
            if line[0] == '>':
                header = line[1:]
            else:
                sequence = line
                fasta[header] = fasta.get(header, '') + sequence
    return fasta


def load_pro_train_data(path1='inter/IMET1.fa', path2='inter/randomIMET1.fa'):

    seqs_pos = _read(path1)
    data = []
    for line in seqs_pos:
        data.append(seqs_pos[line])
    seqs_pos = data

    seqs_neg = _read(path2)
    data = []
    for line in seqs_neg:
        data.append(seqs_neg[line])
    seqs_neg = data

    label_pos = len(seqs_pos) * [1]
    label_neg = len(seqs_neg) * [0]

    tr_data = seqs_pos + seqs_neg
    tr_label = label_pos + label_neg

    train_dataset = _get_onehot(tr_data)

    train_dataset = np.reshape(train_dataset, (train_dataset.shape[0], 1, train_dataset.shape[1], train_dataset.shape[2]))

    return train_dataset, np.array(tr_label)


def load_pro_data(path='inter/IMET1.fa'):

    seqs = _read(path)
    data = []
    for line in seqs:
        data.append(seqs[line])

    dataset = _get_onehot(data)

    dataset = np.reshape(dataset, (dataset.shape[0], 1, dataset.shape[1], dataset.shape[2]))

    return dataset


def load_pro_posdata(path='inter/CCMP525.fa'):

    seqs_pos = _read(path)
    data = []
    for line in seqs_pos:
        data.append(seqs_pos[line])

    label_pos = len(seqs_pos) * [1]

    train_data = _get_onehot(data)
    train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]))

    return train_data, np.array(label_pos)



def _conv_block(ip, nb_filter, dropout_rate=None):
    x = Activation('relu')(ip)
    x = Convolution2D(filters=nb_filter, kernel_size=(4, 4), kernel_initializer="he_uniform", padding="same", use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def _transition_block(ip, nb_filter, dropout_rate=None):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    nb_filter = int(nb_filter / 2)
    x = Convolution2D(filters=nb_filter, kernel_size=(1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False)(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    x = BatchNormalization(mode=0, axis=concat_axis)(x)
    return x, nb_filter


def _dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    feature_list = [x]
    for i in range(nb_layers):
        x = _conv_block(x, growth_rate, dropout_rate)
        feature_list.append(x)
        x = concatenate(feature_list, axis=concat_axis)
        nb_filter += growth_rate
    return x, nb_filter


def _channel_attention(input_feature, ratio=8):
    channel = input_feature._keras_shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)

    avg_pool = Reshape((1, 1, channel))(avg_pool)
    max_pool = Reshape((1, 1, channel))(max_pool)

    avg_pool = shared_layer_one(avg_pool)
    max_pool = shared_layer_one(max_pool)

    avg_pool = shared_layer_two(avg_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def _spatial_attention(input_feature):
    kernel_size = 4

    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def _cbam_block(cbam_feature, ratio=8):
    cbam_feature = _channel_attention(cbam_feature, ratio)
    cbam_feature = _spatial_attention(cbam_feature)
    return cbam_feature


def _create_dense_net(dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=64, dropout_rate=None, verbose=True, linear=False):
    model_input = Input(shape=dim)
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    nb_layers = int((depth - 4) / 3)
    permute_layer = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)))
    x = permute_layer(model_input)
    x = Convolution2D(filters=64, kernel_size=(8, 4), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False, dtype='int64')(x)
    for block_idx in range(nb_dense_block - 1):
        x = _cbam_block(x)
        x, nb_filter = _dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)
        x = _cbam_block(x)
        x, nb_filter = _transition_block(x, nb_filter, dropout_rate=dropout_rate)
    x = _cbam_block(x)
    x, nb_filter = _dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)
    x = _cbam_block(x)
    x = AveragePooling2D((25, 1), strides=(25, 1))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization(mode=0, axis=concat_axis, name='outs')(x)
    if linear:
        x = Dense(1, activation='linear')(x)
    else:
        x = Dense(1, activation='sigmoid')(x)

    densenet = Model(input=model_input, output=x, name="create_dense_net")
    if verbose:
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet


def load_pro_predictor(keras_model_weights="adnppro_trained_model/CCMP525.h5", linear=False):

    predictor = _create_dense_net((1, 1000, 4), depth=34, nb_dense_block=3, growth_rate=12, nb_filter=64, dropout_rate=0.5, verbose=True, linear=linear)
    predictor.load_weights(keras_model_weights)
    predictor.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['accuracy'])

    return predictor
