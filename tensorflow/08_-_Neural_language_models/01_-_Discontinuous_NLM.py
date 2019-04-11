import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

tokens = [
        'i like it'.split(' '),
        'i hate it'.split(' '),
        'i don\'t hate it'.split(' '),
        'i don\'t like it'.split(' '),
    ]

token_prefixes = [ sent[:i] for sent in tokens for i in range(0,len(sent)+1) ]

vocab = ['EDGE']+sorted({ token for sent in tokens for token in sent })
token2index = { token: index for (index, token) in enumerate(vocab) }
indexes = [ [ token2index[token] for token in ['EDGE']+sent+['EDGE'] ] for sent in tokens ]

index_prefixes = [ sent[:i] for sent in indexes for i in range(1,len(sent)) ]
index_prefix_lens = [ len(prefix) for prefix in index_prefixes ]
index_targets  = [ sent[i] for sent in indexes for i in range(1,len(sent)) ]

max_prefix_len = max(index_prefix_lens)
padded_index_prefixes = np.zeros([len(index_prefixes), max_prefix_len])
for (i, prefix) in enumerate(index_prefixes):
    padded_index_prefixes[i, :len(prefix)] = prefix

embedding_size = 2
state_size = 2

class Cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self):
        super(Cell, self).__init__()
        self.W = None
        self.b = None

    @property
    def state_size(self):
        return state_size

    @property
    def output_size(self):
        return state_size

    def build(self, inputs_shape):
        self.W = self.add_variable('W', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=1.0, seed=0))
        self.b = self.add_variable('b', [state_size], tf.float32, tf.zeros_initializer())
        self.built = True
        
    def call(self, x, curr_state):
        state_input = tf.concat([ curr_state, x ], axis=1)
        new_state = tf.tanh(tf.matmul(state_input, self.W) + self.b)
        return (new_state, new_state)

g = tf.Graph()
with g.as_default():
    prefixes = tf.placeholder(tf.int32, [None, None], 'prefixes')
    prefix_lens = tf.placeholder(tf.int32, [None], 'prefix_lens')
    targets = tf.placeholder(tf.int32, [None], 'targets')
    
    batch_size = tf.shape(prefixes)[0]
    
    embedding_matrix = tf.get_variable('embedding_matrix', [ len(vocab), embedding_size ], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    embedded = tf.nn.embedding_lookup(embedding_matrix, prefixes)
    
    init_state = tf.get_variable('init_state', [ state_size ], tf.float32, tf.random_normal_initializer(stddev=1.0, seed=0))
    batch_init = tf.tile(tf.reshape(init_state, [1, state_size]), [batch_size, 1])
    
    cell = Cell()
    (_, state) = tf.nn.dynamic_rnn(cell, embedded, sequence_length=prefix_lens, initial_state=batch_init)

    W = tf.get_variable('W', [state_size, len(vocab)], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    b = tf.get_variable('b', [len(vocab)], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(state, W) + b
    probs = tf.nn.softmax(logits)

    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    
    step = tf.train.AdamOptimizer(0.01).minimize(error)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 1)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 2000+1):
            s.run([ step ], { prefixes: padded_index_prefixes, prefix_lens: index_prefix_lens, targets: index_targets })

            [ train_error ] = s.run([ error ], { prefixes: padded_index_prefixes, prefix_lens: index_prefix_lens, targets: index_targets })
            train_errors.append(train_error)
            
            if epoch%100 == 0:
                print(epoch, train_error)

                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax.set_xlim(0, 2000)
                ax.set_xlabel('epoch')
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel('XE') #Cross entropy
                ax.grid(True)
                ax.set_title('Error progress')
                ax.legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()
        
        [ curr_probs ] = s.run([ probs ], { prefixes: padded_index_prefixes, prefix_lens: index_prefix_lens })
        for (prefix, token_probs) in zip(token_prefixes, curr_probs.tolist()):
            top_predictions = sorted(zip(token_probs, vocab), reverse=True)[:3]
            print(' '.join(prefix))
            print(*[ (token, round(prob, 2)) for (prob, token) in top_predictions ], sep=' | ')
            print()
        
        fig.show()

