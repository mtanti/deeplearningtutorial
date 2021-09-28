import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

token_sents = [
        'i like it <PAD>'.split(' '),  #positive
        'i hate it <PAD>'.split(' '),  #negative
        'i don\'t hate it'.split(' '), #positive
        'i don\'t like it'.split(' '), #negative
    ]
sentiments = [
        [ 1 ],
        [ 0 ],
        [ 1 ],
        [ 0 ]
    ]

vocab = sorted({ token for sent in token_sents for token in sent })

token2index = { token: index for (index, token) in enumerate(vocab) }
index_sents = [ [ token2index[token] for token in sent ] for sent in token_sents ]

token_prefixes = sorted({ tuple(sent[:i]) for sent in token_sents for i in range(len(token_sents)) })
index_prefixes = [ [ token2index[token] for token in token_prefix ] for token_prefix in token_prefixes ]

###################################

class Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, embed_size, state_size, init_stddev):
        super().__init__()
        self.W = None
        self.b = None
        self._embed_size = embed_size
        self._state_size = state_size
        self._init_stddev = init_stddev

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def build(self, inputs_shape):
        self.W = self.add_variable('W', [self._state_size+self._embed_size, self._state_size], tf.float32, tf.random_normal_initializer(stddev=self._init_stddev))
        self.b = self.add_variable('b', [self._state_size], tf.float32, tf.zeros_initializer())
        self.built = True

    def call(self, x, curr_state):
        layer_input = tf.concat([ curr_state, x ], axis=1)
        new_state = tf.tanh(tf.matmul(layer_input, self.W) + self.b)
        return (new_state, new_state) #Return the state as both output and state.

###################################

class Model(object):

    def __init__(self, vocab_size):
        learning_rate = 0.1
        momentum = 0.9
        init_stddev = 1e-1
        embed_size = 2
        state_size = 2 #RNN state vector size.

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sents   = tf.placeholder(tf.int32, [None, None], 'sents')
            self.targets = tf.placeholder(tf.float32, [None, 1], 'targets')

            self.params = []

            batch_size = tf.shape(self.sents)[0]

            with tf.variable_scope('embeddings'):
                self.embedding_matrix = tf.get_variable('embedding_matrix', [ vocab_size, embed_size ], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.params.extend([ self.embedding_matrix ])

                embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.sents)

            with tf.variable_scope('hidden'):
                init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev)) #Allow initial state to be a variable that is optimised with the other parameters
                batch_init = tf.tile(tf.reshape(init_state, [1, state_size]), [batch_size, 1]) #Replicate the initial state for every item in the batch
                self.params.extend([ init_state ])

                cell = Cell(embed_size, state_size, init_stddev)
                (_, self.states) = tf.nn.dynamic_rnn(cell, embedded, initial_state=batch_init)
                self.params.extend([ cell.W, cell.b ])

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [state_size, 1], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])

                logits = tf.matmul(self.states, W) + b
                self.probs = tf.sigmoid(logits)

            self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=logits))

            self.optimiser_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.error)

            self.init = tf.global_variables_initializer()

            self.graph.finalize()

            self.sess = tf.Session()

    def initialise(self):
        return self.sess.run([ self.init ], { })

    def close(self):
        self.sess.close()

    def optimisation_step(self, sents, targets):
        return self.sess.run([ self.optimiser_step ], { self.sents: sents, self.targets: targets })

    def get_params(self):
        return self.sess.run(self.params, { })

    def get_error(self, sents, targets):
        return self.sess.run([ self.error ], { self.sents: sents, self.targets: targets })[0]

    def predict(self, sents):
        return self.sess.run([ self.probs ], { self.sents: sents })[0]

    def get_state(self, sents):
        return self.sess.run([ self.states ], { self.sents: sents })[0]

###################################

max_epochs = 1000

(fig, axs) = plt.subplots(1, 3)

prefix_plots = list()
prefix_texts = list()
for token_prefix in token_prefixes:
    [ prefix_plot ] = axs[0].plot([ 0 ], [ 0 ], linestyle='', marker='o', markersize=10)
    prefix_plots.append(prefix_plot)
    prefix_text = axs[0].text(0, 0, ' '.join(token_prefix), fontdict={ 'fontsize': 8 })
    prefix_texts.append(prefix_text)
axs[0].set_xlim(-1.0, 1.0)
axs[0].set_xlabel('d0')
axs[0].set_ylim(-1.0, 1.0)
axs[0].set_ylabel('d1')
axs[0].grid(True)
axs[0].set_title('Prefixes')

sent_plots = list()
sent_texts = list()
for token_sent in token_sents:
    [ sent_plot ] = axs[1].plot([ 0 ], [ 0 ], linestyle='', marker='o', markersize=10)
    sent_plots.append(sent_plot)
    sent_text = axs[1].text(0, 0, ' '.join(token_sent), fontdict={ 'fontsize': 8 })
    sent_texts.append(sent_text)
axs[1].set_xlim(-1.0, 1.0)
axs[1].set_xlabel('d0')
axs[1].set_ylim(-1.0, 1.0)
axs[1].set_ylabel('d1')
axs[1].grid(True)
axs[1].set_title('Sents')

[ train_error_plot ] = axs[2].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[2].set_xlim(0, max_epochs)
axs[2].set_xlabel('epoch')
axs[2].set_ylim(0.0, 2.0)
axs[2].set_ylabel('XE')
axs[2].grid(True)
axs[2].set_title('Error progress')
axs[2].legend()

fig.tight_layout()
fig.show()

###################################

model = Model(len(vocab))
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(index_sents, sentiments)
    train_errors.append(train_error)

    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')

        #Getting the state of each prefix cannot be done for all prefixes in one batch because they have different lengths and so their list of indexes cannot be made into a matrix (we will fix this in the next section).
        for (prefix_plot, prefix_text, index_prefix) in zip(prefix_plots, prefix_texts, index_prefixes):
            state = model.get_state([index_prefix])[0]
            prefix_plot.set_data([ state[0] ], [ state[1] ])
            prefix_text.set_position( (state[0], state[1]) )

        states = model.get_state(index_sents)
        for (sent_plot, sent_text, state) in zip(sent_plots, sent_texts, states.tolist()):
            sent_plot.set_data([ state[0] ], [ state[1] ])
            sent_text.set_position( (state[0], state[1]) )
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()

    model.optimisation_step(index_sents, sentiments)

print()
print('prefix', 'vector', sep='\t')
for (token_prefix, index_prefix) in zip(token_prefixes, index_prefixes):
    state = model.get_state([index_prefix])[0]
    print(' '.join(token_prefix), np.round(state, 3), sep='\t')
print()
print('sent', 'vector', sep='\t')
states = model.get_state(index_sents)
for (token_sent, state) in zip(token_sents, states.tolist()):
    print(' '.join(token_sent), np.round(state, 3), sep='\t')
print()

probs = model.predict(index_sents)
print('sent', 'sentiment', sep='\t')
for (token_sent, prob) in zip(token_sents, probs.tolist()):
    print(' '.join(token_sent), np.round(prob[0], 3), sep='\t')

model.close()