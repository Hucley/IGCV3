# coding=utf-8
"""
IGCV3.

As described in https://arxiv.org/abs/1804.06202v1

  IGCV2: Interleaved Structured Sparse Convolutional Neural Networks
  

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict

import functools
import tensorflow as tf

slim = tf.contrib.slim

def Print_Activations(net):
    print(net.op.name,' ',net.get_shape().as_list())

batch_norm_params = {
    'center':False,
    'scale': True,
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
}
  
def permutation(inputs,groups=2,scope = 'channel_permutation'):
    # the same operation with channel_permutation
    net = inputs
    N, H, W, C =net.get_shape().as_list()
    assert 0 == C % groups, "Input channels must be a multiple of groups"
    with tf.variable_scope(scope):
        net_reshaped = tf.reshape(net,[-1, H, W, groups, C // groups]) # 先合并重组
        net_transposed = tf.transpose(net_reshaped, [0, 1, 2, 4, 3])  # 转置
        net = tf.reshape(net_transposed, [-1, H, W, C])  # 摊平
        
        return net 
    
def Conv_2d(inputs,num_channel,ksize,stride,scope='Conv'):
        net = inputs
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            net = slim.conv2d(net,num_channel,
                            [ksize,ksize],
                            stride = stride,
                            activation_fn=tf.nn.relu6,
                            scope = scope)
                            
            return net
        
def group_conv2d(inputs,num_channel,groups=2,relu=True,scope = 'Gconv'):
        net = inputs
        input_channels = net.get_shape().as_list()[-1]
        assert 0 == input_channels % groups, "Input channels must be a multiple of num_channel"
        #branches = input_channels // groups 
        num_channels_per_group = num_channel // groups
        with tf.variable_scope(scope):
            net = tf.split(net,groups,axis = 3,name = 'split')
            splits_out =[]
            for i in range(groups):
                end_point = 'Gconv_%d'%i
                temp = slim.conv2d(net[i],
                            num_channels_per_group,
                            [1, 1],
                            stride=1,
                            activation_fn=tf.nn.relu6 if relu else None,
                            scope = end_point)
                splits_out.append(temp)
            net = tf.concat(splits_out, axis=3, name='concat')
  
            return net 
            
            
                                
    
def IGCV3_Block(inputs,num_channel,up_sample,stride,repeat,g1 = 2, g2 = 2, scope = 'igcv3_block'):
    net = inputs
    # print('+++++++++++++++++++++++++')
    # Print_Activations(inputs)
    if repeat <= 0:
        raise ValueError('repeat value of IGCV_Block should be greater than zero.')
    for i in range(repeat):
        start_scope = scope+'_%d'% i
        with tf.variable_scope(start_scope):
            with slim.arg_scope([slim.conv2d,slim.separable_conv2d], padding='SAME'):
                prev_output = net
                net = group_conv2d(net,
                                up_sample*net.get_shape().as_list()[-1],
                                groups = g1,
                                relu = False,
                                scope = 'gconv_expend')
                net = permutation(net, g1,'PM_1')
                net = slim.separable_conv2d(net, None, 
                                [3, 3],
                                stride = stride,
                                depth_multiplier=1,
                                activation_fn=tf.nn.relu6,
                                scope='dpwise')                
                net = group_conv2d(net,
                                num_channel,
                                groups = g2,
                                relu = False,
                                scope = 'gconv_linear')
                net = permutation(net, num_channel//g2,'PM_2')
                
                if stride == 1:
                    if prev_output.get_shape().as_list()[-1] != net.get_shape().as_list()[-1]: 
                        prev_output = slim.conv2d(prev_output, num_channel, [1, 1],
                                                activation_fn=None,
                                                biases_initializer=None,
                                                scope='res_match')
                    net = tf.add(prev_output, net)                   
        
                return net



def IGCV_v3(inputs,
            is_training=True,
            embedding_size=128,
            reuse=None,
            scope=None):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'IGCV3', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = inputs                                           #224x224
            net = Conv_2d(net,32,3,2,scope='igcv_conv3x3')             #112x112
            net = IGCV3_Block(net,16,1,1,1,scope = 'igcv_v3_1')     
            net = IGCV3_Block(net,24,6,2,1,scope = 'igcv_v3_2')    #56x56
            net = IGCV3_Block(net,24,6,1,3,scope = 'igcv_v3_3')
            net = IGCV3_Block(net,32,6,2,1,scope = 'igcv_v3_4')    #28x28
            net = IGCV3_Block(net,32,6,1,5,scope = 'igcv_v3_5')
            net = IGCV3_Block(net,64,6,2,1,scope = 'igcv_v3_6')    #14x14
            net = IGCV3_Block(net,64,6,1,7,scope = 'igcv_v3_7')
            net = IGCV3_Block(net,96,6,1,1,scope = 'igcv_v3_8')
            net = IGCV3_Block(net,96,6,1,5,scope = 'igcv_v3_9')
            net = IGCV3_Block(net,160,6,2,1,scope = 'igcv_v3_10')  #7x7
            net = IGCV3_Block(net,160,6,1,5,scope = 'igcv_v3_11')
            net = IGCV3_Block(net,320,6,1,1,scope = 'igcv_v3_12')
            net = Conv_2d(net,1280,1,1,scope = 'igcv_conv1x1')
            net = slim.avg_pool2d(net, [7,7],padding='VALID',scope='gloabl_avg_pool')
            net = slim.flatten(net)
            if embedding_size:
                net = slim.fully_connected(net, embedding_size, activation_fn=None,
                                               scope='FC', reuse=False)

            return net


def IGCV_arg_scope(weight_decay=0.0005,use_batch_norm=True):
    """Defines the default arg scope.
    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
    Returns:
      An `arg_scope` to use for the IGCV model.
    """

    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
    #weights_initializer = slim.xavier_initializer_conv2d()
    regularizer = slim.l2_regularizer(weight_decay)
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, slim.fully_connected], 
                        weights_initializer=weights_initializer,
                        weights_regularizer=regularizer,
                        biases_regularizer=regularizer):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
                        activation_fn=tf.nn.relu6,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        biases_initializer=None) as sc:
                    return sc
                    
def inference(inputs,
              embedding_size=128,
              keep_probability=0.999,
              weight_decay=0.0005,
              use_batch_norm=False,
              phase_train=True,
              reuse=None,
              scope='IGCV3'):
    arg_scope = IGCV_arg_scope(weight_decay,use_batch_norm)
    with slim.arg_scope(arg_scope):
        net = IGCV_v3(inputs,
                        is_training=phase_train,
                        embedding_size=embedding_size,
                        reuse=reuse,
                        scope=scope)
    return net, None
