from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def feed_foward_network(inputs, reuse=False, is_training=True):

    with tf.variable_scope("feed_foward_network"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, weights_initializer=tf.glorot_uniform_initializer()): 
            net = slim.fully_connected(inputs, 512)
            net = slim.fully_connected(net, 256)
            net = slim.fully_connected(net, 256)
            net = slim.fully_connected(net, 512)
            net = slim.fully_connected(net, 1024)
        net = slim.fully_connected(net, 161, weights_initializer=tf.glorot_uniform_initializer())
        return net

def prediction_network(inputs, reuse=False):

    with tf.variable_scope("prediction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        inputs = slim.flatten(inputs)
        net_a = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(inputs, 512, activation_fn=tf.nn.tanh)

        net_a = slim.fully_connected(net_a, 512, activation_fn=tf.nn.tanh)
        net_b = slim.fully_connected(net_b, 512, activation_fn=tf.nn.tanh)

        spectrum_a = slim.fully_connected(net_a, 101, activation_fn=tf.nn.tanh)
        spectrum_b = slim.fully_connected(net_b, 101, activation_fn=tf.nn.tanh)

        spectra = tf.concat([spectrum_a, spectrum_b], axis=1)
        spectra = tf.expand_dims(spectra, axis=2)
        return spectra


def recognition_network(feature, spectra, latent_dims, reuse=False):

    with tf.variable_scope("recognition_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        spectra = slim.flatten(spectra)
        feature = slim.flatten(feature)
        spectrum_a, spectrum_b = spectra[:,:101], spectra[:,101:]
        net = tf.concat([feature, spectrum_a, spectrum_b], axis=1)
        mean = slim.fully_connected(net, latent_dims, activation_fn=tf.nn.tanh)
        covariance = slim.fully_connected(net, latent_dims, activation_fn=tf.nn.tanh)

        return mean, covariance

def reconstruction_network(spectra, latent_variables, reuse=False):

    with tf.variable_scope("reconstruction_network"):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        spectra = slim.flatten(spectra)
        spectrum_a, spectrum_b = spectra[:,:101], spectra[:,101:]
        net = tf.concat([spectrum_a, spectrum_b, latent_variables], axis=1)
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.tanh) # 512
        net = slim.fully_connected(net, 512, activation_fn=tf.nn.tanh) # 512
        net = slim.fully_connected(net, 8*8*256, activation_fn=tf.nn.tanh) # 8*8*256
        net = tf.reshape(net, [-1,8,8,256])
        net = slim.conv2d_transpose(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.leaky_relu) # (16, 16, 128)
        net = slim.conv2d_transpose(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.leaky_relu) # (32, 32, 64)
        net = slim.conv2d_transpose(inputs=net, num_outputs=1, kernel_size=[3, 3], stride=2, activation_fn=tf.nn.tanh)  # (64, 64, 1)

        return net

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target): # mae??? not mse??
    return tf.reduce_mean(tf.abs(in_-target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
