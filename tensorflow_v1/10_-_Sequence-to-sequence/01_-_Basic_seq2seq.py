import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

max_epochs = 5000
init_stddev = 0.0001
source_embedding_size = 2
target_embedding_size = 2
source_state_size = 2
target_state_size = 2
max_seq_len = 10

source_tokens = [
        'i like it'.split(' '),
        'i hate it'.split(' '),
        'i don\'t hate it'.split(' '),
        'i don\'t like it'.split(' '),
    ]
target_tokens = [
        'i don\'t like it'.split(' '),
        'i don\'t hate it'.split(' '),
        'i hate it'.split(' '),
        'i like it'.split(' '),
    ]

source_vocab = [ 'EDGE' ] + sorted({ token for sent in source_tokens for token in sent })
source_token2index = { token: index for (index, token) in enumerate(source_vocab) }
source_index2token = { index: token for (index, token) in enumerate(source_vocab) }
source_max_len = max(len(sent) for sent in source_tokens)

index_source_indexes = []
index_source_lens = []
for sent in source_tokens:
    source_lens = len(sent)
    source_index = [ source_token2index[token] for token in sent ] + [ 0 for _ in range(source_max_len - source_lens) ]
    
    index_source_lens.append(source_lens)
    index_source_indexes.append(source_index)

target_vocab = [ 'EDGE' ] + sorted({ token for sent in target_tokens for token in sent })
target_token2index = { token: index for (index, token) in enumerate(target_vocab) }
target_index2token = { index: token for (index, token) in enumerate(target_vocab) }
target_max_len = max(len(sent) for sent in target_tokens) + 1 #Plus edge token

index_target_prefixes = []
index_target_lens = []
index_target_targets = []
for sent in target_tokens:
    target_len = len(sent) + 1 #Plus edge token
    target_index = [ target_token2index[token] for token in sent ]
    target_prefix = [ target_token2index['EDGE'] ] + target_index + [ 0 for _ in range(target_max_len - target_len) ]
    target_target = target_index + [ target_token2index['EDGE'] ] + [ 0 for _ in range(target_max_len - target_len) ]
    
    index_target_prefixes.append(target_prefix)
    index_target_lens.append(target_len)
    index_target_targets.append(target_target)

g = tf.Graph()
with g.as_default():
    source_indexes = tf.placeholder(tf.int32, [None, None], 'source_indexes')
    source_lens = tf.placeholder(tf.int32, [None], 'source_lens')
    target_prefixes = tf.placeholder(tf.int32, [None, None], 'target_prefixes')
    target_lens = tf.placeholder(tf.int32, [None], 'target_lens')
    target_targets = tf.placeholder(tf.int32, [None, None], 'target_targets')
    
    batch_size = tf.shape(source_indexes)[0]
    target_seq_width = tf.shape(target_prefixes)[1]
    
    with tf.variable_scope('source'):
        with tf.variable_scope('embedding'):
            embedding_matrix = tf.get_variable('embedding_matrix', [len(source_vocab), source_embedding_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
            embedded = tf.nn.embedding_lookup(embedding_matrix, source_indexes)
        
        with tf.variable_scope('init_state'):
            init_state = tf.get_variable('init_state', [source_state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
            batch_init = tf.tile(tf.reshape(init_state, [1, source_state_size]), [batch_size, 1])
        
        with tf.variable_scope('rnn'):
            cell = tf.contrib.rnn.GRUCell(source_state_size)
            (_, states) = tf.nn.dynamic_rnn(cell, embedded, sequence_length=source_lens, initial_state=batch_init)
        
        #Encode source sentence vector using an RNN
        source_encoding = states
    
    with tf.variable_scope('targets'):
        with tf.variable_scope('embedding'):
            embedding_matrix = tf.get_variable('embedding_matrix', [len(target_vocab), target_embedding_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
            embedded = tf.nn.embedding_lookup(embedding_matrix, target_prefixes)
        
        #Condition language model using source sentence encoding via init-inject
        with tf.variable_scope('init_state'):
            W = tf.get_variable('W', [source_state_size, target_state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
            b = tf.get_variable('b', [target_state_size], tf.float32, tf.zeros_initializer())
            init_states = tf.matmul(source_encoding, W) + b
            
            batch_init = init_states
        
        with tf.variable_scope('rnn'):
            cell = tf.contrib.rnn.GRUCell(target_state_size)
            (outputs, _) = tf.nn.dynamic_rnn(cell, embedded, sequence_length=target_lens, initial_state=batch_init)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [target_state_size, len(target_vocab)], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
            b = tf.get_variable('b', [len(target_vocab)], tf.float32, tf.zeros_initializer())
            outputs_2d = tf.reshape(outputs, [batch_size*target_seq_width, target_state_size])
            logits_2d = tf.matmul(outputs_2d, W) + b
            logits = tf.reshape(logits_2d, [batch_size, target_seq_width, len(target_vocab)])
            probs = tf.nn.softmax(logits)
        
            next_word_probs = probs[:, -1, :]

    mask = tf.sequence_mask(target_lens, target_seq_width, tf.float32)
    error = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_targets, logits=logits)*mask)/tf.cast(tf.reduce_sum(target_lens), tf.float32)
    
    step = tf.train.AdamOptimizer().minimize(error)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 1)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error', sep='\t')
        for epoch in range(1, max_epochs+1):
            s.run([ step ], { source_indexes: index_source_indexes, source_lens: index_source_lens, target_prefixes: index_target_prefixes, target_lens: index_target_lens, target_targets: index_target_targets })
    
            [ train_error ] = s.run([ error ], { source_indexes: index_source_indexes, source_lens: index_source_lens, target_prefixes: index_target_prefixes, target_lens: index_target_lens, target_targets: index_target_targets })
            train_errors.append(train_error)
            
            if epoch%100 == 0:
                print(epoch, train_error, sep='\t')

                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax.set_xlim(0, max_epochs)
                ax.set_xlabel('epoch')
                ax.set_ylim(0.0, 2.0)
                ax.set_ylabel('XE') #Cross entropy
                ax.grid(True)
                ax.set_title('Error progress')
                ax.legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()

        for sent in source_tokens:
            source = [ source_token2index[token] for token in sent ]
            prefix_prob = 1.0
            index_prefix = [ target_token2index['EDGE'] ]
            for _ in range(max_seq_len):
                [ curr_probs ] = s.run([ next_word_probs ], { source_indexes: [ source ], source_lens: [ len(source) ], target_prefixes: [ index_prefix ], target_lens: [ len(index_prefix) ] })
                selected_index = np.argmax(curr_probs[0, :]) #Greedy search
                prefix_prob = prefix_prob*curr_probs[0, selected_index]
                
                index_prefix.append(selected_index)
                
                if selected_index == target_token2index['EDGE']:
                    break
            generated = [ target_index2token[i] for i in index_prefix[1:-1] ]

            print('Input sentence:    ', ' '.join(sent))
            print('Generated sentence:', ' '.join(generated))
            print('Sentence probability:', prefix_prob)
            print()
        
        fig.show()