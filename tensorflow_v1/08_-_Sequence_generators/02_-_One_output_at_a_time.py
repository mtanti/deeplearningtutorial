import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

training_sequence = [ 0, 1, 0, 1 ]

###################################

class Model(object):

    def __init__(self, vocab_size):
        init_stddev = 0.001
        embed_size = 2
        state_size = 2

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.seq_len = tf.placeholder(tf.int32, [], 'seq_len')
            self.target = tf.placeholder(tf.int32, [None], 'target')

            self.params = []
            self.rnn_initialisers = []

            with tf.variable_scope('hidden'):
                input_vector = tf.get_variable('input_vector', [embed_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                input_seq = tf.tile(tf.reshape(input_vector, [1, 1, embed_size]), [1, self.seq_len, 1])
                self.params.extend([ input_vector ])

                self.init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                batch_init = tf.reshape(self.init_state, [1, state_size])
                self.params.extend([ self.init_state ])

                cell = tf.contrib.rnn.BasicRNNCell(state_size, tf.tanh)
                (outputs, self.state) = tf.nn.dynamic_rnn(cell, input_seq, initial_state=batch_init)
                [ W, b ] = cell.weights
                self.rnn_initialisers.extend([
                        tf.assign(W, tf.random_normal([state_size+embed_size, state_size], stddev=init_stddev)),
                        tf.assign(b, tf.zeros([state_size]))
                    ])
                self.params.extend([ W, b ])

            with tf.variable_scope('output'):
                out_W = tf.get_variable('W', [state_size, vocab_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                out_b = tf.get_variable('b', [vocab_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ out_W, out_b ])

                outputs_2d = tf.reshape(outputs, [1*self.seq_len, state_size])
                logits = tf.matmul(outputs_2d, out_W) + out_b
                self.probs = tf.nn.softmax(logits)

            with tf.variable_scope('onestep'):
                #This is a separate part of the neural network that is only used at test time and only uses variables that are already defined above.
                self.onestep_state = tf.placeholder(tf.float32, [state_size], 'onestep_state') #The state needs to be supplied externally.
                onestep_batch_state = tf.reshape(self.onestep_state, [1, state_size])
                onestep_batch_input = tf.reshape(input_vector, [1, embed_size])
                (onestep_new_state, _) = cell.call(onestep_batch_input, onestep_batch_state) #The supplied state is passed to a single time step call of the RNN.

                self.onestep_new_state = onestep_new_state[0, :]
                self.onestep_probs = tf.nn.softmax(tf.matmul(onestep_new_state, out_W) + out_b)[0, :]

            self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=logits))

            self.optimiser_step = tf.train.AdamOptimizer().minimize(self.error)

            self.init = tf.global_variables_initializer()

            self.graph.finalize()

            self.sess = tf.Session()

    def initialise(self):
        self.sess.run([ self.init ], { })
        self.sess.run(self.rnn_initialisers, { })

    def close(self):
        self.sess.close()

    def optimisation_step(self, seq_len, target):
        return self.sess.run([ self.optimiser_step ], { self.seq_len: seq_len, self.target: target })

    def get_params(self):
        return self.sess.run(self.params, { })

    def get_error(self, seq_len, target):
        return self.sess.run([ self.error ], { self.seq_len: seq_len, self.target: target })[0]

    def predict(self, seq_len):
        return self.sess.run([ self.probs ], { self.seq_len: seq_len })[0]

    def get_state(self, seq_len):
        return self.sess.run([ self.state ], { self.seq_len: seq_len })[0]

    def get_init_state(self):
        return self.sess.run([ self.init_state ], { })[0]

    def onestep_predict(self, state):
        return self.sess.run([ self.onestep_probs, self.onestep_new_state ], { self.onestep_state: state })

###################################

max_epochs = 2000

(fig, ax) = plt.subplots(1, 1)

[ train_error_plot ] = ax.plot([], [], color='red', linestyle='-', linewidth=1, label='train')
ax.set_xlim(0, max_epochs)
ax.set_xlabel('epoch')
ax.set_ylim(0.0, 2.0)
ax.set_ylabel('XE')
ax.grid(True)
ax.set_title('Error progress')
ax.legend()

fig.tight_layout()
fig.show()

###################################

model = Model(2)
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(len(training_sequence), training_sequence)
    train_errors.append(train_error)

    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()

    model.optimisation_step(len(training_sequence), training_sequence)

print()
state = model.get_init_state()
print('Generated sequence')
for _ in range(20):
    [ curr_probs, state ] = model.onestep_predict(state)
    print(np.argmax(curr_probs))

model.close()
