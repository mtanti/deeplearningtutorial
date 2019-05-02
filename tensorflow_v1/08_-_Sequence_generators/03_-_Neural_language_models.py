import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

max_epochs = 6000
init_stddev = 0.0001
embedding_size = 2
state_size = 2
max_seq_len = 10

tokens = [
        'i like it'.split(' '),
        'i hate it'.split(' '),
        'i don\'t hate it'.split(' '),
        'i don\'t like it'.split(' '),
    ]

sent_lens = [ len(sent) for sent in tokens ]
max_sent_len = max(sent_lens)

vocab = [ 'EDGE' ] + sorted({ token for sent in tokens for token in sent }) #Add the EDGE token to the vocabulary which is used to mark the beginning or end of a sentence (it is recommended that you sort a set into a list rather than just convert it so that the list is always in the same order)
token2index = { token: index for (index, token) in enumerate(vocab) }
index2token = { index: token for (index, token) in enumerate(vocab) }
index_prefixes = []
index_prefix_lens = []
index_targets = []
for sent in tokens:
    sent = [ 'EDGE' ] + sent + [ 'EDGE' ] #Add the EDGE token to the beginning and end of every sentence
    for i in range(1, len(sent)):
        prefix = [ token2index[token] for token in sent[:i] ] #Prefix of sentence
        prefix_len = len(prefix) #Length fo prefix (without padding)
        padded_prefix = prefix + [ 0 for _ in range((max_sent_len + 1) - prefix_len) ] #Prefix padded to have length equal to maximum length prefix
        target = token2index[sent[i]] #Next word after prefix
        
        index_prefixes.append(padded_prefix)
        index_prefix_lens.append(prefix_len)
        index_targets.append(target)

g = tf.Graph()
with g.as_default():
    prefixes = tf.placeholder(tf.int32, [None, None], 'prefixes')
    prefix_lens = tf.placeholder(tf.int32, [None], 'prefix_lens')
    targets = tf.placeholder(tf.int32, [None], 'targets')
    
    batch_size = tf.shape(prefixes)[0]
    
    embedding_matrix = tf.get_variable('embedding_matrix', [len(vocab), embedding_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    embedded = tf.nn.embedding_lookup(embedding_matrix, prefixes)
    
    init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    batch_init = tf.tile(tf.reshape(init_state, [1, state_size]), [batch_size, 1])
    
    cell = tf.contrib.rnn.BasicRNNCell(state_size, tf.tanh)
    (_, state) = tf.nn.dynamic_rnn(cell, embedded, sequence_length=prefix_lens, initial_state=batch_init)

    W = tf.get_variable('W', [state_size, len(vocab)], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    b = tf.get_variable('b', [len(vocab)], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(state, W) + b
    probs = tf.nn.softmax(logits)

    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    
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
            s.run([ step ], { prefixes: index_prefixes, prefix_lens: index_prefix_lens, targets: index_targets })

            [ train_error ] = s.run([ error ], { prefixes: index_prefixes, prefix_lens: index_prefix_lens, targets: index_targets })
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

        print('Sampling random sentence...')
        print()
        
        prefix_prob = 1.0
        index_prefix = [ token2index['EDGE'] ] #Start with the edge token, which has a 100% probability of being the first token
        for _ in range(max_seq_len): #For a maximum of 10 words in the sentence, generate a sentence one word at a time
            [ curr_probs ] = s.run([ probs ], { prefixes: [ index_prefix ], prefix_lens: [ len(index_prefix) ] })
            selected_index = np.random.choice(range(len(vocab)), p=curr_probs[0, :]) #Sample a token index from the vocabulary according to the probabilities given by the neural network
            prefix_prob = prefix_prob*curr_probs[0, selected_index]
            
            top3_tokens = sorted(zip(curr_probs[0, :].tolist(), vocab), reverse=True)[:3]
            print('Prefix:           ', ' '.join([ index2token[i] for i in index_prefix ]))
            print('Top 3 next tokens:', ', '.join([ '{} ({:.3})'.format(token, prob) for (prob, token) in top3_tokens ]))
            print('Selected token:   ', index2token[selected_index])
            print('New probability:  ', prefix_prob)
            print()
            
            index_prefix.append(selected_index)
            
            if selected_index == token2index['EDGE']:
                break

        print('Generated sentence:', ' '.join([ index2token[i] for i in index_prefix[1:-1] ])) #Show generated sentence without the edge tokens
        print('Sentence probability:', prefix_prob)
        
        fig.show()