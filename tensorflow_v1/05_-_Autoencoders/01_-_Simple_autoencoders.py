import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

#Training set consisting of one-hot vectors.
train_x = [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ]

###################################

class Model(object):

    def __init__(self):
        learning_rate = 0.1
        momentum = 0.9
        init_stddev = 1e-2
        hidden_layer_size = 2 #Hidden layer is a vector that is smaller than the input in order to force compression.

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None, 4], 'xs') #Note that we do not need to use target values as they are the same as the input values.
            
            self.params = []

            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', [4, hidden_layer_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [hidden_layer_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.thought_vector = tf.sigmoid(tf.matmul(self.xs, W) + b)

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [hidden_layer_size, 4], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [4], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                logits = tf.matmul(self.thought_vector, W) + b
                self.ys = tf.sigmoid(logits)
            
            self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.xs, logits=logits)) #Use the inputs as target values.
            
            self.optimiser_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.error)

            self.init = tf.global_variables_initializer()
            
            self.graph.finalize()

            self.sess = tf.Session()
    
    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, xs):
        return self.sess.run([ self.optimiser_step ], { self.xs: xs })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_error(self, xs):
        return self.sess.run([ self.error ], { self.xs: xs })[0]
    
    def encode(self, xs):
        return self.sess.run([ self.thought_vector ], { self.xs: xs })[0]
    
    def decode(self, thoughtvecs):
        return self.sess.run([ self.ys ], { self.thought_vector: thoughtvecs })[0]
    
    def predict(self, xs):
        return self.sess.run([ self.ys ], { self.xs: xs })[0]

###################################

max_epochs = 3000

(fig, axs) = plt.subplots(1, 2)

thoughtvec_plots = list()
thoughtvec_texts = list()
for x in train_x:
    [ thoughtvec_plot ] = axs[0].plot([ 0 ], [ 0 ], linestyle='', marker='o', markersize=10)
    thoughtvec_plots.append(thoughtvec_plot)
    thoughtvec_text = axs[0].text(0, 0, '{}{}{}{}'.format(*x), fontdict={ 'fontsize': 8 })
    thoughtvec_texts.append(thoughtvec_text)
axs[0].set_xlim(0.0, 1.0)
axs[0].set_xlabel('d0')
axs[0].set_ylim(0.0, 1.0)
axs[0].set_ylabel('d1')
axs[0].grid(True)
axs[0].set_title('Thought vectors')

[ train_error_plot ] = axs[1].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[1].set_xlim(0, max_epochs)
axs[1].set_xlabel('epoch')
axs[1].set_ylim(0.0, 0.26)
axs[1].set_ylabel('XE')
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
    train_error = model.get_error(train_x)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        
        thoughtvecs = model.encode(train_x)
        
        for (thoughtvec_plot, thoughtvec_text, thoughtvec) in zip(thoughtvec_plots, thoughtvec_texts, thoughtvecs.tolist()):
            thoughtvec_plot.set_data([ thoughtvec[0] ], [ thoughtvec[1] ])
            thoughtvec_text.set_position( (thoughtvec[0], thoughtvec[1]) )
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(train_x)
print()

thoughtvecs = model.encode(train_x)
ys = model.predict(train_x)
print('x', 'thought', 'y', sep='\t')
for (x, thoughtvec, y) in zip(train_x, thoughtvecs.tolist(), ys):
    print(np.array(x), np.round(thoughtvec, 2), np.round(y, 2), sep='\t')
print()

thoughtvecs = np.array([[0,0], [0,1], [1,0], [1,1]])
ys = model.decode(thoughtvecs)
print('randthought', 'y', sep='\t')
for (thoughtvec, y) in zip(thoughtvecs.tolist(), ys):
    print(np.array(thoughtvec), np.round(y, 2), sep='\t')

model.close()