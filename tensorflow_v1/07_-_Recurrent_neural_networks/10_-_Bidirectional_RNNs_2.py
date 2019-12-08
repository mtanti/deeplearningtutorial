import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

token_sents = [
        'i like it'.split(' '),        #positive
        'i hate it'.split(' '),        #negative
        'i don\'t hate it'.split(' '), #positive
        'i don\'t like it'.split(' '), #negative
    ]
sent_lens = [
        3,
        3,
        4,
        4,
    ]
sentiments = [
        [ 1 ],
        [ 0 ],
        [ 1 ],
        [ 0 ]
    ]

vocab = sorted({ token for sent in token_sents for token in sent })
max_len = max(sent_lens)

token2index = { token: index for (index, token) in enumerate(vocab) }
index_sents = [ [ token2index[token] for token in sent ] + [ 0 for _ in range(max_len - len(sent)) ] for sent in token_sents ] #Add zeros to the end of each sentence so that all sentences are equal to the maximum length (can be some other index instead of zero).

token_prefixes = sorted({ tuple(sent[:i]) for sent in token_sents for i in range(len(token_sents)) })
prefix_lens = [ len(prefix) for prefix in token_prefixes ]
max_prefix_len = max(prefix_lens)
index_prefixes = [ [ token2index[token] for token in prefix ] + [ 0 for _ in range(max_prefix_len - len(prefix)) ] for prefix in token_prefixes ]

###################################

class Model(object):

    def __init__(self, vocab_size):
        init_stddev = 1e-1
        embed_size = 2
        state_size = 2
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sents     = tf.placeholder(tf.int32, [None, None], 'sents')
            self.sent_lens = tf.placeholder(tf.int32, [None], 'sent_lens')
            self.targets   = tf.placeholder(tf.float32, [None, 1], 'targets')

            self.params = []

            batch_size = tf.shape(self.sents)[0]
            
            with tf.variable_scope('embeddings'):
                self.embedding_matrix = tf.get_variable('embedding_matrix', [ vocab_size, embed_size ], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.params.extend([ self.embedding_matrix ])

                embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.sents)

            with tf.variable_scope('hidden'):
                init_state_fw = tf.get_variable('init_state_fw', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                init_state_bw = tf.get_variable('init_state_bw', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                
                batch_init_fw = tf.tile(tf.reshape(init_state_fw, [1, state_size]), [batch_size, 1])
                batch_init_bw = tf.tile(tf.reshape(init_state_bw, [1, state_size]), [batch_size, 1])
                
                cell_fw = tf.contrib.rnn.BasicRNNCell(2, tf.tanh)
                cell_bw = tf.contrib.rnn.BasicRNNCell(2, tf.tanh)
                
                (_, (self.state_fw, self.state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedded, sequence_length=self.sent_lens, initial_state_fw=batch_init_fw, initial_state_bw=batch_init_bw)
                self.states = tf.concat([ self.state_fw, self.state_bw ], axis=1)
                [ W_fw, b_fw ] = cell_fw.weights
                [ W_bw, b_bw ] = cell_bw.weights
                self.rnn_initialisers = [
                        tf.assign(W_fw, tf.random_normal([state_size+embed_size, state_size], stddev=init_stddev)),
                        tf.assign(b_bw, tf.zeros([state_size])),
                        
                        tf.assign(W_fw, tf.random_normal([state_size+embed_size, state_size], stddev=init_stddev)),
                        tf.assign(b_bw, tf.zeros([state_size]))
                    ]
                self.params.extend([ W_fw, b_fw, W_bw, b_bw ])

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [2*state_size, 1], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                
                logits = tf.matmul(self.states, W) + b
                self.probs = tf.sigmoid(logits)
            
            self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=logits))
            
            self.optimiser_step = tf.train.AdamOptimizer().minimize(self.error)
        
            self.init = tf.global_variables_initializer()
            
            self.graph.finalize()

            self.sess = tf.Session()
    
    def initialise(self):
        self.sess.run([ self.init ], { })
        self.sess.run(self.rnn_initialisers, { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, sents, sent_lens, targets):
        return self.sess.run([ self.optimiser_step ], { self.sents: sents, self.sent_lens: sent_lens, self.targets: targets })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_error(self, sents, sent_lens, targets):
        return self.sess.run([ self.error ], { self.sents: sents, self.sent_lens: sent_lens, self.targets: targets })[0]
    
    def predict(self, sents, sent_lens):
        return self.sess.run([ self.probs ], { self.sents: sents, self.sent_lens: sent_lens })[0]
    
    def get_state(self, sents, sent_lens):
        return self.sess.run([ self.states ], { self.sents: sents, self.sent_lens: sent_lens })[0]

###################################

max_epochs = 2000

(fig, ax) = plt.subplots(1, )

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

model = Model(len(vocab))
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(index_sents, sent_lens, sentiments)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(index_sents, sent_lens, sentiments)

print()
probs = model.predict(index_sents, sent_lens)
print('sent', 'sentiment', sep='\t')
for (token_sent, prob) in zip(token_sents, probs.tolist()):
    print(' '.join(token_sent), np.round(prob[0], 3), sep='\t')
    
model.close()