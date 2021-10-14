       # -*- c  odi   ng: utf-8 -   *-
"""
Created on Fri Nov  2 23:37:36 2018

@author: hp
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
import sys
def windows(nrows, size):
    start,step = 0, 2
    while start < nrows:
        yield start, start + size
        start += step

def segment_signal(features,labels,window_size = 15):
    segments = np.empty((0,window_size))
    segment_labels = np.empty((0))
    nrows = len(features)
    for(start, end) in windows(nrows,window_size) :
        #print(start,end)
        if(len(data.iloc[start:end]) == window_size):
            segment = features[start:end].T  #Transpose to get segment of size 5 x 15 
            label = labels[(end-1)]
            segments = np.vstack([segments,segment]) 
            segment_labels = np.append(segment_labels,label)
    segments = segments.reshape(-1,5,window_size,1) # number of features  = 5
    segment_labels = segment_labels.reshape(-1,1)
    return segments,segment_labels

data = pd.read_csv("BatteryData.csv") 
features = scale(data.iloc[:,0:5]) # select required columns and scale them
labels = data.iloc[:,5] # select RUL 

segments, labels = segment_signal(features,labels)
#print(segments, labels)
train_test_split = np.random.rand(len(segments)) < 0.70
train_x = segments[train_test_split]
train_y = labels[train_test_split]
test_x = segments[~train_test_split]
test_y = labels[~train_test_split]
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def apply_conv(x,kernel_height,kernel_width,num_channels,depth):
    weights = weight_variable([kernel_height, kernel_width, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights,[1,1,1,1],padding="VALID"),biases))
    
def apply_max_pool(x,kernel_height,kernel_width,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1], strides=[1, 1, stride_size, 1], padding = "VALID")



num_labels = 1
batch_size = 10
num_hidden = 800
learning_rate = 0.0001
training_epochs =30
input_height = 5
input_width = 15
num_channels = 1
total_batches = train_x.shape[0] // batch_size

X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

c = apply_conv(X, kernel_height = 5, kernel_width = 4, num_channels = 1, depth = 8) 
p = apply_max_pool(c,kernel_height = 1, kernel_width = 2, stride_size = 2) 
c = apply_conv(p, kernel_height = 1, kernel_width = 3, num_channels = 8, depth = 14) 
p = apply_max_pool(c,kernel_height = 1, kernel_width = 2, stride_size = 2) 

shape = p.get_shape().as_list()
flat = tf.reshape(p, [-1, shape[1] * shape[2] * shape[3]])
f_weights = weight_variable([shape[1] * shape[2] * shape[3], num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(flat, f_weights),f_biases))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.add(tf.matmul(f, out_weights),out_biases)

cost_function = tf.reduce_mean(tf.square(y_- Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

with tf.Session() as session: 
    tf.global_variables_initializer().run()
    print("Training set RMSE")
    for epoch in range(training_epochs):
        for b in range(total_batches):    
            offset = (b * batch_size) % (train_x.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size),:]
            _, c = session.run([optimizer, cost_function],feed_dict={X: batch_x, Y : batch_y})
            
        p_tr = session.run(y_, feed_dict={X:  train_x})
        tr_mse = tf.sqrt(tf.reduce_mean(tf.square(p_tr - train_y)))
        print("Training set RMSE: %.4f" % session.run(tr_mse))
        #print("Training set Score: %.4f" %(r2_score(train_y,p_tr)))
    p_ts = session.run(y_, feed_dict={X:  test_x})
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    #print(p_ts)
    sys.stdout = orig_stdout
    f.close()
    print(test_y, p_ts) 
    
    ts_mse = tf.sqrt(tf.reduce_mean(tf.square(p_ts - test_y)))
    #print("Test set Score: %.4f" %(r2_score(test_y,p_ts)))
    print("Test set RMSE: %.4f" % session.run(ts_mse)) 
     