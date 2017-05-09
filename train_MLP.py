"""
With this script you can train a basic MLP
Specify the configuration settings at the beginning according to your 
problem.

This script was written for TensorFlow 1.0 

Author: Xavier Faure
contact: xavier.faure42(at)gmail.com
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from MLP import MLP
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""

# Input params
image_width = 28
image_height = 28
image_channel = 3

# Path to the trainings and validation set
train_file = '../data-notMNIST/train.txt'
val_file = '../data-notMNIST/validation.txt'

# Learning params
learning_rate = 0.0001
num_epochs = 20
batch_size = 256

# Network params
num_classes = 10
hidden_layer_size = 100


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, image_width, image_height, image_channel])
y = tf.placeholder(tf.float32, [None, num_classes])


# Initialize model
model = MLP(x, image_width * image_height * image_channel, num_classes, hidden_layer_size)

# Link variable to model output
score = model.fc3

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

# Train op
with tf.name_scope("train"):
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.minimize(loss)


  
# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
 
# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, 
                                     horizontal_flip = False, shuffle = True,
                                     scale_size=(image_width, image_height),
                                     nb_classes=num_classes)
val_generator = ImageDataGenerator(val_file, shuffle = False, scale_size=(image_width, image_height), nb_classes=num_classes)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)


# Start Tensorflow session
with tf.Session() as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
 
  print("{} Start training...".format(datetime.now()))
  # Loop over number of epochs
  for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch+1)) 
        step = 1
        while step < train_batches_per_epoch:
            
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, 
                                          y: batch_ys})
            step += 1
            
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx, 
                                                y: batch_ty})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {}".format(datetime.now(), test_acc))
        
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()