import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

#Training set for a binary decoder (convert binary number into index).
train_x = [ [0,0], [0,1], [1,0], [1,1] ]
train_y = [   0,     1,     2,     3   ]

###################################

class Model(object):

    def __init__(self):
        learning_rate = 5.0
        init_stddev = 0.3
        hidden_layer_size = 2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None, 2], 'xs')
            self.ts = tf.placeholder(tf.int32, [None], 'ts') #Targets are integer scalars that say which index of the softmax should be 1.
            
            self.params = []

            W = tf.get_variable('W', [2, 1], tf.float32, tf.zeros_initializer())
            b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
            self.params.extend([ W, b ])
            self.ys = tf.sigmoid(tf.matmul(self.xs, W) + b)
            
            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', [2, hidden_layer_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [hidden_layer_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.hs = tf.sigmoid(tf.matmul(self.xs, W) + b)

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [hidden_layer_size, 4], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [4], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                logits = tf.matmul(self.hs, W) + b
                self.ys = tf.nn.softmax(logits) #Softmax function transforms the logits into a probability distribution.
            
            #Taking the mean of the categorical cross entropy error of each training set item.
            self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ts, logits=logits))
            
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

max_epochs = 2000

(fig1, axs1) = plt.subplots(2, 2)
(fig2, ax2)  = plt.subplots(1, 1)

(x0s, x1s) = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
xs = np.stack([ np.reshape(x0s, [50*50]), np.reshape(x1s, [50*50]) ], axis=1)

output_imgs = list()
index = 0
for row in range(2):
    for col in range(2):
        output_img = axs1[row,col].matshow(np.full_like(x0s, 0.5), vmin=0.0, vmax=1.0, extent=(0.0,1.0,0.0,1.0), cmap='bwr')
        output_imgs.append(output_img)
        axs1[row,col].set_xlabel('x0')
        axs1[row,col].set_ylabel('x1')
        axs1[row,col].set_title('Output {}'.format(index))
        axs1[row,col].grid(True)
        
        index += 1

[ train_error_plot ] = ax2.plot([], [], color='red', linestyle='-', linewidth=1, label='train')
ax2.set_xlim(0, max_epochs)
ax2.set_xlabel('epoch')
ax2.set_ylim(0.0, 0.26)
ax2.set_ylabel('XE')
ax2.grid(True)
ax2.set_title('Error progress')
ax2.legend()

fig1.tight_layout()
fig1.show()

fig2.tight_layout()
fig2.show()

###################################

model = Model()
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(train_x, train_y)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        
        ys = model.predict(xs)
        ys = np.reshape(ys, [50, 50, 4])
        
        for (index, output_img) in enumerate(output_imgs):
            output_img.set_data(ys[:,:,index])
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig1.canvas.flush_events()
        fig2.canvas.flush_events()
    
    model.optimisation_step(train_x, train_y)

print()
print('00:', np.round(model.predict([[0,0]])[0], 3))
print('01:', np.round(model.predict([[0,1]])[0], 3))
print('10:', np.round(model.predict([[1,0]])[0], 3))
print('11:', np.round(model.predict([[1,1]])[0], 3))

model.close()