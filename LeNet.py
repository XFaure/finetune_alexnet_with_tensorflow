"""
This is a Tensorflow implementation of LeNet by Yann LeCun 
http://yann.lecun.com/exdb/lenet/

Author: Xavier Faure
contact : xavier.faure42(at)gmail.com
"""
import tensorflow as tf
import numpy as np

class LeNet(object):
  
  def __init__(self, x, num_classes,):
    
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes

    # Call the create function to build the computational graph of AlexNet
    self.create()
    
  def create(self):
    
    # 1st Layer: Conv (w ReLu) -> Pool
    conv1 = conv(self.X, 5, 5, 6, 1, 1, padding = 'SAME', name = 'conv1')    
    pool1 = max_pool(conv1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')
    
    # 2nd Layer: Conv (w ReLu) -> Pool -
    conv2 = conv(pool1, 5, 5, 16, 1, 1, padding = 'VALID',name = 'conv2')
    pool2 = max_pool(conv2, 2, 2, 2, 2, padding = 'VALID', name ='pool2')
    
    # 3rd Layer: FC + Relu
    flattened = tf.reshape(pool2, [-1, 400])
    fc1 = fc(flattened, 400, 120, name='fc1')
       
    # 3rd Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    self.fc2 = fc(fc1, 120, self.NUM_CLASSES, relu = False, name='fc2')
 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
  """
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    
    if groups == 1:
      conv = convolve(x, weights)
      
    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      
      # Concat the convolved output together again
      conv = tf.concat(axis = 3, values = output_groups)
      
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)
        
    return relu
    
def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)
 
# Fully connected definition
def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    