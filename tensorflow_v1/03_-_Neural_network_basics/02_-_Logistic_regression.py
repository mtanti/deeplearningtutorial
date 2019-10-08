import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

#The training set (currently set for an OR function).
train_x = [ [0,0], [0,1], [1,0], [1,1] ]
train_y = [  [0],   [1],   [1],   [1]  ]

###################################

class Model(object):

    def __init__(self):
        learning_rate = 1.0

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None, 2], 'xs')
            self.ts = tf.placeholder(tf.float32, [None, 1], 'ts')
            
            self.params = []

            W = tf.get_variable('W', [2, 1], tf.float32, tf.zeros_initializer())
            b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
            self.params.extend([ W, b ])
            self.ys = tf.sigmoid(tf.matmul(self.xs, W) + b)
            
            self.error = tf.reduce_mean((self.ys - self.ts)**2)
            
            self.optimiser_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.error)

            self.init = tf.global_variables_initializer()
            
            self.graph.finalize()

            self.sess = tf.Session()
    
    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, xs, ts):
        return self.sess.run([ self.optimiser_step ], { self.xs: xs, self.ts: ts })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_error(self, xs, ts):
        return self.sess.run([ self.error ], { self.xs: xs, self.ts: ts })[0]
    
    def predict(self, xs):
        return self.sess.run([ self.ys ], { self.xs: xs })[0]

###################################

max_epochs = 500

(fig, axs) = plt.subplots(1, 2)

(x0s, x1s) = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50)) #Get how x0 and x1 for every grid square on the heat map.
xs = np.stack([ np.reshape(x0s, [50*50]), np.reshape(x1s, [50*50]) ], axis=1) #Combine the x0s and x1s into a list of coordinates.

output_img = axs[0].matshow(np.full_like(x0s, 0.5), vmin=0.0, vmax=1.0, extent=(0.0,1.0,0.0,1.0), cmap='bwr') #Start heat map with just 0.5 everywhere.
axs[0].set_xlabel('x0')
axs[0].set_ylabel('x1')
axs[0].set_title('Output')
axs[0].grid(True)

[ train_error_plot ] = axs[1].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[1].set_xlim(0, max_epochs)
axs[1].set_xlabel('epoch')
axs[1].set_ylim(0.0, 0.26)
axs[1].set_ylabel('MSE')
axs[1].grid(True)
axs[1].set_title('Error progress')
axs[1].legend()

fig.tight_layout()
fig.show()

###################################

model = Model()
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(train_x, train_y)
    train_errors.append(train_error)
    
    if epoch%20 == 0:
        print(epoch, train_error, sep='\t')
        
        ys = model.predict(xs)
        ys = np.reshape(ys, [50, 50]) #Reshape list of coordinates into a grid.
        
        output_img.set_data(ys)
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(train_x, train_y)

model.close()