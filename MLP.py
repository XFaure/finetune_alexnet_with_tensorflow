"""
This is a basic MLP definition

This script was written for TensorFlow 1.0 

Author: Xavier Faure
contact: xavier.faure42(at)gmail.com
"""
import tensorflow as tf
import numpy as np

class MLP(object):
  
  def __init__(self, x, input_size, num_classes, hidden_layer_size):
    
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.INPUT_SIZE = input_size
    self.HIDDEN_LAYER_SIZE = hidden_layer_size

    # Call the create function to build the computational graph of AlexNet
    self.create()
    
  def create(self):
    
    # 1st Layer: FC + Relu
    flattened = tf.reshape(self.X, [-1, self.INPUT_SIZE])
    fc1 = fc(flattened, self.INPUT_SIZE, self.HIDDEN_LAYER_SIZE, name='fc1')
    
    # 2nd Layer: FC + Relu
    fc2 = fc(fc1, self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE, name = 'fc2')
    
    # 3rd Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    self.fc3 = fc(fc2, self.HIDDEN_LAYER_SIZE, self.NUM_CLASSES, relu = False, name='fc3')
 
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
    