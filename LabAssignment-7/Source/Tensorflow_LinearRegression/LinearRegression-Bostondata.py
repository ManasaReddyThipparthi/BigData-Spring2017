from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
rng = np.random
boston = load_boston()

print('Number of rows and columns on Boston DataSet:'+str(boston.data.shape))

print('Keys in Boston Dataset'+str(boston.keys()))

print('Features in Boston Dataset'+str(boston.feature_names))

print('Description of Boston Description'+boston.DESCR)
"""
Attributes:13 Continuous variables,1 Binary valued attribute
['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']"""


features = np.array(boston.data)
bos = pd.DataFrame(boston.data)
trX = np.array(bos[5])
trY = np.array(boston.target)

# create symbolic variables
X = tf.placeholder("float")
Y = tf.placeholder("float")

# create a shared variable for the weight matrix
w = tf.Variable(rng.randn(), name="weights")
b = tf.Variable(rng.randn(), name="bias")

# prediction function
y_model = tf.add(tf.multiply(X, w), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(y_model-Y, 2))/(2*trX.shape[0])

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializing the variables
init = tf.global_variables_initializer()

# you need to initialize variables
sess.run(init)
for i in range(20):
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={X: trX, Y: trY})

print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')

# Testing or Inference
test_X = np.asarray([rng.randn(),rng.randn()])
test_Y = sess.run(w)*test_X + sess.run(b)
print("Testing... (Mean square loss Comparison)")

testing_cost = sess.run(
    tf.reduce_sum(tf.pow(y_model - Y, 2)) / (2 * test_X.shape[0]),
    feed_dict={X: test_X, Y: test_Y}) # same function as cost above
print("Testing cost=", testing_cost)
print("Absolute mean square loss difference:", abs(
    training_cost - testing_cost))

plt.plot(trX, trY,'ro', label='Original data')
plt.plot(trX, sess.run(w) * trX + sess.run(b), label='Fitted Line')
plt.legend(loc='upper left')
plt.ylabel('Price of House')
plt.xlabel('Average number of rooms per dwelling')
plt.show()