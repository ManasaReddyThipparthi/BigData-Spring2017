from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
rng = np.random

trX = np.linspace(-2.00, 2.10, 101)
#pyplot.plot([0.95577, 0.00, 8.140])


# create a y value which is approximately linear but with some random noise

trY = 2 * trX + 4+np.random.randn(*trX.shape) * 0.033

# create symbolic variables

X = tf.placeholder("float")

Y = tf.placeholder("float")

# create a shared variable for the weight matrix

w = tf.Variable(rng.randn(), name="weights")

b = tf.Variable(rng.randn(), name="bias")

# prediction function
y_model = tf.add(tf.mul(X, w), b)


# Mean squared error

cost = tf.reduce_sum(tf.pow(y_model-Y, 2))/(2*100)

# construct an optimizer to minimize cost and fit line to my data

train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)


# Launch the graph in a session
sess = tf.Session()

# Initializing the variables

init = tf.initialize_all_variables()


# you need to initialize variables
sess.run(init)


for i in range(100):
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})
        #fit = np.polyfit(x, y, 1)
        #fit_fn = np.poly1d(fit)
        #plt.plot(x, y, 'yo', x, fit_fn(x), '--k')

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={X: trX, Y: trY})

print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')
#plt.show()


points = np.arange(training_cost, sess.run(w), sess.run(b))
points = np.arange(1, 2, 3)
dx, dy = np.meshgrid(points, points)
z = (np.sin(dx)+np.sin(dy))
plt.imshow(z)
plt.colorbar()
#plt.title('plot for sin(x)+sin(y)')
plt.show()



#Testing or Inference
test_X = np.asarray([rng.randn(),rng.randn()])

test_Y = 2*test_X + 4

print("Testing... (Mean square loss Comparison)")

testing_cost = sess.run(
    tf.reduce_sum(tf.pow(y_model - Y, 2)) / (2 * test_X.shape[0]),
    feed_dict={X: test_X, Y: test_Y})  # same function as cost above
print("Testing cost=", testing_cost)
print("Absolute mean square loss difference:", abs(
    training_cost - testing_cost))