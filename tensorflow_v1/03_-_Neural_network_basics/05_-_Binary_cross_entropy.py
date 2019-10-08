import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

###################################

class Model(object):

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None], 'xs')
            self.ts = tf.placeholder(tf.float32, [None], 'ts')
            
            self.ys = tf.sigmoid(self.xs)
            
            self.square_error = (self.ys - self.ts)**2
            
            binary_cross_entropy_0 = -tf.log(1 - self.ys) #Error function when the target output is 0.
            binary_cross_entropy_1 = -tf.log(self.ys) #Error function when the target output is 1.
            self.binary_cross_entropy = binary_cross_entropy_1*self.ts + binary_cross_entropy_0*(1 - self.ts) #General error function for any target output.
            
            self.graph.finalize()

            self.sess = tf.Session()
    
    def close(self):
        self.sess.close()
    
    def get_square_errors(self, xs, ts):
        return self.sess.run([ self.square_error ], { self.xs: xs, self.ts: ts })[0] #Get a list of errors, one for every value of x.
    
    def get_binary_cross_entropies(self, xs, ts):
        return self.sess.run([ self.binary_cross_entropy ], { self.xs: xs, self.ts: ts })[0] #Get a list of errors, one for every value of x.
    
    def predict(self, xs):
        return self.sess.run([ self.ys ], { self.xs: xs })[0]

###################################

model = Model()

(fig, axs) = plt.subplots(2, 2)

xs = np.linspace(-10, 10, 50)
ys = model.predict(xs)

ts = np.zeros_like(xs)
square_error_0 = model.get_square_errors(xs, ts)
binary_cross_entropy_0 = model.get_binary_cross_entropies(xs, ts)

ts = np.ones_like(xs)
square_error_1 = model.get_square_errors(xs, ts)
binary_cross_entropy_1 = model.get_binary_cross_entropies(xs, ts)

axs[0,0].plot(xs, ys, color='red', linestyle='-', linewidth=1, label='sigmoid')
axs[0,0].plot(xs, square_error_0, color='blue', linestyle=':', linewidth=3, label='square_error')
axs[0,0].set_xlim(-10, 10)
axs[0,0].set_xlabel('x')
axs[0,0].set_ylim(-0.1, 5.0)
axs[0,0].set_ylabel('y')
axs[0,0].set_title('Square error (t=0)')
axs[0,0].legend()
axs[0,0].grid(True)

axs[0,1].plot(xs, ys, color='red', linestyle='-', linewidth=1, label='sigmoid')
axs[0,1].plot(xs, binary_cross_entropy_0, color='blue', linestyle=':', linewidth=3, label='binary_cross_entropy')
axs[0,1].set_xlim(-10, 10)
axs[0,1].set_xlabel('x')
axs[0,1].set_ylim(-0.1, 5.0)
axs[0,1].set_ylabel('y')
axs[0,1].set_title('Binary cross entropy (t=0)')
axs[0,1].legend()
axs[0,1].grid(True)

axs[1,0].plot(xs, ys, color='red', linestyle='-', linewidth=1, label='sigmoid')
axs[1,0].plot(xs, square_error_1, color='blue', linestyle=':', linewidth=3, label='square_error')
axs[1,0].set_xlim(-10, 10)
axs[1,0].set_xlabel('x')
axs[1,0].set_ylim(-0.1, 5.0)
axs[1,0].set_ylabel('y')
axs[1,0].set_title('Square error (t=1)')
axs[1,0].legend()
axs[1,0].grid(True)

axs[1,1].plot(xs, ys, color='red', linestyle='-', linewidth=1, label='sigmoid')
axs[1,1].plot(xs, binary_cross_entropy_1, color='blue', linestyle=':', linewidth=3, label='binary_cross_entropy')
axs[1,1].set_xlim(-10, 10)
axs[1,1].set_xlabel('x')
axs[1,1].set_ylim(-0.1, 5.0)
axs[1,1].set_ylabel('y')
axs[1,1].set_title('Binary cross entropy (t=1)')
axs[1,1].legend()
axs[1,1].grid(True)

fig.tight_layout()
fig.show()

model.close()