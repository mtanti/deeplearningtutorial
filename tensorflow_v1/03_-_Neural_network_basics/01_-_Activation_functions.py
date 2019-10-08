import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

###################################

class Model(object):

    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None], 'xs')
            
            #The 4 of the most common activation functions.
            self.ys_sig = tf.sigmoid(self.xs)
            self.ys_tanh = tf.tanh(self.xs)
            self.ys_relu = tf.nn.relu(self.xs)
            self.ys_softmax = tf.nn.softmax(self.xs)
            
            graph.finalize()
            
            self.sess = tf.Session()
    
    def predict_sig(self, xs):
        return self.sess.run([ self.ys_sig ], { self.xs: xs })[0]
    
    def predict_tanh(self, xs):
        return self.sess.run([ self.ys_tanh ], { self.xs: xs })[0]
    
    def predict_relu(self, xs):
        return self.sess.run([ self.ys_relu ], { self.xs: xs })[0]
    
    def predict_softmax(self, xs):
        return self.sess.run([ self.ys_softmax ], { self.xs: xs })[0]

###################################

model = Model()

(fig, axs) = plt.subplots(1, 3)

xs = np.linspace(-10, 10, 50)

#Sigmoid
ys = model.predict_sig(xs)

axs[0].plot(xs, ys, color='blue', linestyle='-', linewidth=3)
axs[0].set_xlim(-10, 10)
axs[0].set_xlabel('x')
axs[0].set_ylim(-2, 2)
axs[0].set_ylabel('y')
axs[0].set_title('sigmoid(xs)')
axs[0].grid(True)

#Hyperbolic tangent
ys = model.predict_tanh(xs)

axs[1].plot(xs, ys, color='blue', linestyle='-', linewidth=3)
axs[1].set_xlim(-10, 10)
axs[1].set_xlabel('x')
axs[1].set_ylim(-2, 2)
axs[1].set_title('tanh(xs)')
axs[1].grid(True)

#Rectified linear unit
ys = model.predict_relu(xs)

axs[2].plot(xs, ys, color='blue', linestyle='-', linewidth=3)
axs[2].set_xlim(-10, 10)
axs[2].set_xlabel('x')
axs[2].set_ylim(-2, 2)
axs[2].set_title('ReLU(xs)')
axs[2].grid(True)

fig.tight_layout()
fig.show()

###########################

(fig, axs) = plt.subplots(1, 2)

xs = np.array([ -1, 0, 1 ], np.float32)

#Softmax
ys = model.predict_softmax(xs)

axs[0].bar(xs, ys)
for (x, y) in zip(xs, np.round(ys, 2)):
    axs[0].annotate(y, xy=(x, y), xytext=(x, y+5), textcoords='offset points', ha='center', va='bottom')
axs[0].set_xlabel('logits')
axs[0].set_ylim(0.0, 1.0)
axs[0].set_title('softmax(xs)')
axs[0].grid(True)

#Softmax with added constant to logits (xs).
#Subtracting 1 from every element in the logits will not change the output probabilities.
ys = model.predict_softmax(xs - 1)

axs[1].bar(xs, ys)
for (x, y) in zip(xs, np.round(ys, 2)):
    axs[1].annotate(y, xy=(x, y), xytext=(x, y+5), textcoords='offset points', ha='center', va='bottom')
axs[1].set_xlabel('logits')
axs[1].set_ylim(0.0, 1.0)
axs[1].set_title('softmax(xs - 1)')
axs[1].grid(True)

fig.tight_layout()
fig.show()

model.close()