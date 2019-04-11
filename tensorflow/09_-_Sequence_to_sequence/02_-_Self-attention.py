import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

tokens = [
        'i like it'.split(' '),
        'i hate it'.split(' '),
        'i don\'t hate it'.split(' '),
        'i don\'t like it'.split(' '),
    ]

vocab = ['EDGE']+sorted({ token for sent in tokens for token in sent })
token2index = { token: index for (index, token) in enumerate(vocab) }
indexes = [ [ token2index[token] for token in sent ] for sent in tokens ]
index_lens = [ len(prefix) for prefix in indexes ]

max_index_len = max(index_lens)
padded_indexes = np.zeros([len(indexes), max_index_len])
for (i, sent) in enumerate(indexes):
    padded_indexes[i, :len(sent)] = sent

index_prefixes = [ [token2index['EDGE']]+sent for sent in indexes ]
index_targets  = [ sent+[token2index['EDGE']] for sent in indexes ]
index_prefix_lens = [ len(prefix) for prefix in index_prefixes ]

max_prefix_len = max(index_prefix_lens)
padded_index_prefixes = np.zeros([len(index_prefixes), max_prefix_len])
padded_index_targets = np.zeros([len(index_prefixes), max_prefix_len])
for (i, (prefix, target)) in enumerate(zip(index_prefixes, index_targets)):
    padded_index_prefixes[i, :len(prefix)] = prefix
    padded_index_targets[i, :len(target)] = target

embedding_size = 4
state_size = 8
attention_size = 4

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
    sources = tf.placeholder(tf.int32, [None, None], 'sources')
    source_lens = tf.placeholder(tf.int32, [None], 'source_lens')
    prefixes = tf.placeholder(tf.int32, [None, None], 'prefixes')
    prefix_lens = tf.placeholder(tf.int32, [None], 'prefix_lens')
    targets = tf.placeholder(tf.int32, [None, None], 'targets')
    
    batch_size = tf.shape(sources)[0]
    num_steps_sources = tf.shape(sources)[1]
    num_steps_prefixes = tf.shape(prefixes)[1]

    with tf.variable_scope('sources'):
        embedding_matrix_sources = tf.get_variable('embedding_matrix_sources', [ len(vocab), embedding_size ], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
        embedded_sources = tf.nn.embedding_lookup(embedding_matrix_sources, sources)
        
        with tf.variable_scope('fw'):
            init_state_fw_sources = tf.get_variable('init_state_fw_sources', [ state_size ], tf.float32, tf.random_normal_initializer(stddev=1.0, seed=0))
            batch_init_fw_sources = tf.tile(tf.reshape(init_state_fw_sources, [1, state_size]), [batch_size, 1])

            cell_fw_sources = Cell()

        with tf.variable_scope('bw'):
            init_state_bw_sources = tf.get_variable('init_state_bw_sources', [ state_size ], tf.float32, tf.random_normal_initializer(stddev=1.0, seed=0))
            batch_init_bw_sources = tf.tile(tf.reshape(init_state_bw_sources, [1, state_size]), [batch_size, 1])

            cell_bw_sources = Cell()
            
        ((outputs_fw_sources, outputs_bw_sources), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw_sources, cell_bw_sources, embedded_sources, sequence_length=source_lens, initial_state_fw=batch_init_fw_sources, initial_state_bw=batch_init_bw_sources)
        pre_outputs_sources = tf.concat([outputs_fw_sources, outputs_bw_sources], axis=2)
        W = tf.get_variable('W', [ 2*state_size, state_size ], tf.float32, tf.random_normal_initializer(stddev=1.0, seed=0))
        outputs_sources_2d = tf.matmul(tf.reshape(pre_outputs_sources, [batch_size*num_steps_sources, 2*state_size]), W)
        outputs_sources = tf.reshape(outputs_sources_2d, [batch_size, num_steps_sources, state_size])

        with tf.variable_scope('attention'):
            W1 = tf.get_variable('W1', [ state_size, attention_size ], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
            pre_attention_2d = tf.tanh(tf.matmul(outputs_sources_2d, W1))

            W2 = tf.get_variable('W2', [ attention_size, 1 ], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
            attention_logits_2d = tf.matmul(pre_attention_2d, W2)
            attention_logits = tf.reshape(attention_logits_2d, [batch_size, num_steps_sources])

            mask = tf.sequence_mask(source_lens, num_steps_sources, tf.float32)
            masked_attention_logits = attention_logits*mask + -1e10*(1 - mask)

            attention = tf.nn.softmax(masked_attention_logits)

            expanded_attention = tf.tile(tf.reshape(attention, [batch_size, num_steps_sources, 1]), [1, 1, state_size])
            attended_sources = tf.reduce_sum(outputs_sources*expanded_attention, axis=1)

    with tf.variable_scope('prefixes'):
        embedding_matrix_prefixes = tf.get_variable('embedding_matrix_prefixes', [ len(vocab), embedding_size ], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
        embedded_prefixes = tf.nn.embedding_lookup(embedding_matrix_prefixes, prefixes)

        #init_state_prefixes = tf.get_variable('init_state_prefixes', [ state_size ], tf.float32, tf.random_normal_initializer(stddev=1.0, seed=0))
        #batch_init_prefixes = tf.tile(tf.reshape(init_state_prefixes, [1, state_size]), [batch_size, 1])
        batch_init_prefixes = attended_sources
        
        cell_prefixes = Cell()
        (outputs_prefixes, _) = tf.nn.dynamic_rnn(cell_prefixes, embedded_prefixes, sequence_length=prefix_lens, initial_state=batch_init_prefixes)

    W = tf.get_variable('W', [state_size, len(vocab)], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    b = tf.get_variable('b', [len(vocab)], tf.float32, tf.zeros_initializer())
    outputs_2d = tf.reshape(outputs_prefixes, [batch_size*num_steps_prefixes, state_size])
    logits_2d = tf.matmul(outputs_2d, W) + b
    logits = tf.reshape(logits_2d, [batch_size, num_steps_prefixes, len(vocab)])
    probs = tf.nn.softmax(logits)

    mask = tf.sequence_mask(prefix_lens, num_steps_prefixes, tf.float32)
    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)*mask)
    
    step = tf.train.AdamOptimizer().minimize(error)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 1)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 5000+1):
            s.run([ step ], { sources: padded_indexes, source_lens: index_lens, prefixes: padded_index_prefixes, prefix_lens: index_prefix_lens, targets: padded_index_targets })

            [ train_error ] = s.run([ error ], { sources: padded_indexes, source_lens: index_lens, prefixes: padded_index_prefixes, prefix_lens: index_prefix_lens, targets: padded_index_targets })
            train_errors.append(train_error)
            
            if epoch%100 == 0:
                print(epoch, train_error)

                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax.set_xlim(0, 5000)
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
        
        [ curr_probs, curr_attention ] = s.run([ probs, attention ], { sources: padded_indexes, source_lens: index_lens, prefixes: padded_index_prefixes, prefix_lens: index_prefix_lens })
        for (sent, sent_probs, sent_attention) in zip(tokens, curr_probs.tolist(), curr_attention.tolist()):
            print('-------')
            print(' '.join(sent))
            print(*[round(a, 2) for a in sent_attention], sep=' ')
            print('-------')
            for i in range(len(sent)+1):
                prefix = sent[:i]
                token_probs = sent_probs[i]
                top_predictions = sorted(zip(token_probs, vocab), reverse=True)[:3]
                print(' '.join(prefix))
                print(*[ (token, round(prob, 2)) for (prob, token) in top_predictions ], sep=' | ')
                print()
        
        fig.show()

