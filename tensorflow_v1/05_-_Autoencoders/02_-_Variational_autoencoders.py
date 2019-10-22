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
        learning_rate = 0.01
        momentum = 0.9
        init_stddev = 1e-2
        thoughtvec_size = 2
        random_normal_weight = 0.3 #The weighting given to making the encoder's distribution equal to a standard normal distribution.

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None, 4], 'xs')
            
            self.params = []

            #Encoder creates a random normal vector using a mean and a standard deviation computed based on the input.
            with tf.variable_scope('encoder'):
                W_mean = tf.get_variable('W_mean', [4, thoughtvec_size], tf.float32, tf.zeros_initializer())
                b_mean = tf.get_variable('b_mean', [thoughtvec_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.thought_mean = tf.matmul(self.xs, W_mean) + b_mean #No need for activation function here.

                W_stddev = tf.get_variable('W_stddev', [4, thoughtvec_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b_stddev = tf.get_variable('b_stddev', [thoughtvec_size], tf.float32, tf.ones_initializer())
                thought_log_stddev = tf.matmul(self.xs, W_stddev) + b_stddev #Standard deviation needs to be positive, so assume that this is the log of the stddev so that it will be used as an exponent of e, thus making it positive.
                self.thought_stddev = tf.exp(thought_log_stddev)

                #Generate the random thought vector.
                self.thought_vector = self.thought_stddev*tf.random_normal(tf.shape(self.thought_mean), 0.0, 1.0) + self.thought_mean #This is the definition of a random normal distribution: standard_deviation*standard_normal() + mean.

            with tf.variable_scope('decoder'):
                W = tf.get_variable('W', [thoughtvec_size, 4], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [4], tf.float32, tf.zeros_initializer())
                logits = tf.matmul(self.thought_vector, W) + b
                self.ys = tf.sigmoid(logits)
            
            self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.xs))
            
            #This is a measure of how far the encoding is from being a standard normal distribution (mean=0, stddev=1).
            kl_divergence = 0.5*tf.reduce_mean(tf.square(self.thought_mean) + tf.square(self.thought_stddev) - 2*thought_log_stddev - 1.0)
            
            #Multi-objective optimisation
            loss = self.error + random_normal_weight*kl_divergence
                    
            self.optimiser_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

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

    def get_thought_means(self, xs):
        return self.sess.run([ self.thought_mean ], { self.xs: xs })[0]

    def get_thought_stddevs(self, xs):
        return self.sess.run([ self.thought_stddev ], { self.xs: xs })[0]

    def decode(self, thoughtvecs):
        return self.sess.run([ self.ys ], { self.thought_vector: thoughtvecs })[0]
    
    def predict(self, xs):
        return self.sess.run([ self.ys ], { self.xs: xs })[0]

###################################

max_epochs = 8000

(fig, axs) = plt.subplots(1, 2)

thoughtvec_plots = list()
thoughtvec_texts = list()
for x in train_x:
    [ thoughtvec_plot ] = axs[0].plot([ 0 ], [ 0 ], linestyle='', marker='o', markersize=10)
    thoughtvec_plots.append(thoughtvec_plot)
    thoughtvec_text = axs[0].text(0, 0, '{}{}{}{}'.format(*x), fontdict={ 'fontsize': 8 })
    thoughtvec_texts.append(thoughtvec_text)
axs[0].set_xlim(-5.0, 5.0)
axs[0].set_xlabel('d0')
axs[0].set_ylim(-5.0, 5.0)
axs[0].set_ylabel('d1')
axs[0].grid(True)
axs[0].set_title('Thought vectors')

[ train_error_plot ] = axs[1].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[1].set_xlim(0, max_epochs)
axs[1].set_xlabel('epoch')
axs[1].set_ylim(0.0, 1.0)
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

means = model.get_thought_means(train_x)
stddevs = model.get_thought_stddevs(train_x)
print('x', 'mean', 'stddev', sep='\t')
for (x, m, s) in zip(train_x, means.tolist(), stddevs.tolist()):
    print(np.array(x), np.round(m, 2), np.round(s, 2), sep='\t')
print()

thoughtvecs = np.random.normal(0.0, 1.0, size=[4,2])
ys = model.decode(thoughtvecs)
print('randthought', 'y', sep='\t')
for (thoughtvec, y) in zip(thoughtvecs.tolist(), ys):
    print(np.round(thoughtvec, 2), np.round(y, 2), sep='\t')

model.close()