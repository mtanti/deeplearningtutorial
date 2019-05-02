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

vocab = [ 'EDGE' ] + sorted({ token for sent in tokens for token in sent })
token2index = { token: index for (index, token) in enumerate(vocab) }
index2token = { index: token for (index, token) in enumerate(vocab) }
index_prefixes = []
index_lens = []
index_targets = []
for sent in tokens:
    sent = [ 'EDGE' ] + sent + [ 'EDGE' ]
    indexes = [ token2index[token] for token in sent ]
    indexes_len = len(indexes) - 1
    prefixes = indexes[:-1] + [ 0 for _ in range((max_sent_len + 1) - indexes_len) ]
    targets = indexes[1:] + [ 0 for _ in range((max_sent_len + 1) - indexes_len) ]
    
    index_prefixes.append(prefixes)
    index_lens.append(indexes_len)
    index_targets.append(targets)

g = tf.Graph()
with g.as_default():
    prefixes = tf.placeholder(tf.int32, [None, None], 'prefixes')
    prefix_lens = tf.placeholder(tf.int32, [None], 'prefix_lens')
    targets = tf.placeholder(tf.int32, [None, None], 'targets')
    
    batch_size = tf.shape(prefixes)[0]
    seq_width = tf.shape(prefixes)[1]
    
    embedding_matrix = tf.get_variable('embedding_matrix', [len(vocab), embedding_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    embedded = tf.nn.embedding_lookup(embedding_matrix, prefixes)
    
    init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    batch_init = tf.tile(tf.reshape(init_state, [1, state_size]), [batch_size, 1])
    
    cell = tf.contrib.rnn.BasicRNNCell(state_size, tf.tanh)
    (outputs, _) = tf.nn.dynamic_rnn(cell, embedded, sequence_length=prefix_lens, initial_state=batch_init)

    W = tf.get_variable('W', [state_size, len(vocab)], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    b = tf.get_variable('b', [len(vocab)], tf.float32, tf.zeros_initializer())
    outputs_2d = tf.reshape(outputs, [batch_size*seq_width, state_size])
    logits_2d = tf.matmul(outputs_2d, W) + b
    logits = tf.reshape(logits_2d, [batch_size, seq_width, len(vocab)])
    probs = tf.nn.softmax(logits)
    
    next_word_probs = probs[:, -1, :]

    mask = tf.sequence_mask(prefix_lens, seq_width, tf.float32)
    error = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)*mask)/tf.cast(tf.reduce_sum(prefix_lens), tf.float32)
    
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
            s.run([ step ], { prefixes: index_prefixes, prefix_lens: index_lens, targets: index_targets })

            [ train_error ] = s.run([ error ], { prefixes: index_prefixes, prefix_lens: index_lens, targets: index_targets })
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

        print('------------')
        print('Generating sentence by greedy search...')
        
        prefix_prob = 1.0 #Probability of full prefix is the product of each word's probability
        index_prefix = [ token2index['EDGE'] ]
        for _ in range(max_seq_len):
            [ curr_probs ] = s.run([ next_word_probs ], { prefixes: [ index_prefix ], prefix_lens: [ len(index_prefix) ] })
            selected_index = np.argmax(curr_probs[0, :]) #Pick the most probable word given the prefix
            prefix_prob = prefix_prob*curr_probs[0, selected_index]
            
            index_prefix.append(selected_index)
            
            top3_tokens = sorted(zip(curr_probs[0, :].tolist(), vocab), reverse=True)[:3]
            print('Prefix:           ', ' '.join([ index2token[i] for i in index_prefix ]))
            print('Top 3 next tokens:', ', '.join([ '{} ({:.3})'.format(token, prob) for (prob, token) in top3_tokens ]))
            print('Selected token:   ', index2token[selected_index])
            print('New probability:  ', prefix_prob)
            print()
            
            if selected_index == token2index['EDGE']:
                break
            
        print('Generated sentence:', ' '.join([ index2token[i] for i in index_prefix[1:-1] ]))
        print('Sentence probability:', prefix_prob)
        
        print()

        print('------------')
        print('Generating sentence by beam search...')
        
        beam_width = 3 #This controls the breadth of exploration (becomes more accurate but also slower as this is made larger)
        beam = [ {
                'prefix_prob': 1.0, #The probability of the prefix
                'is_complete': False, #Whether the prefix is a complete sentence
                'index_prefix': [ token2index['EDGE'] ] #The prefix
            } ]
        for _ in range(max_seq_len):
            new_beam = []
            for beam_entry in beam:
                if beam_entry['is_complete'] == True: #If the prefix is complete then just copy it over to the new beam
                    new_beam.append(beam_entry)
                else: #If not then expand the prefix to every possible next token
                    [ curr_probs ] = s.run([ next_word_probs ], { prefixes: [ beam_entry['index_prefix'] ], prefix_lens: [ len(beam_entry['index_prefix']) ] })
                    for (index, token_prob) in zip(range(len(vocab)), curr_probs[0, :].tolist()):
                        new_beam.append({
                                'prefix_prob': beam_entry['prefix_prob']*token_prob,
                                'is_complete': index == token2index['EDGE'],
                                'index_prefix': beam_entry['index_prefix'] + [ index ]
                            })
            #Keep only the top 'beam_width' most probable prefixes in the beam
            new_beam.sort(key=lambda beam_entry:(beam_entry['prefix_prob'], beam_entry['is_complete']), reverse=True) #In case two prefixes have the same probability, give priority to the complete sentence
            beam = new_beam[:beam_width]
            
            print('Beam contents:')
            for beam_entry in beam:
                print('', ' '.join([ index2token[i] for i in beam_entry['index_prefix'] ]), '({:.3})'.format(beam_entry['prefix_prob']))
            print()
            
            #As soon a the most probable prefix in the beam is a complete sentence, stop searching and return it
            if beam[0]['is_complete'] == True:
                break

        print('Generated sentence:', ' '.join([ index2token[i] for i in beam[0]['index_prefix'][1:-1] ]))
        print('Sentence probability:', beam[0]['prefix_prob'])
        
        '''
        Tips for more efficient performance:
        1) Use a heap queue (import heapq) instead of a list to store beams entries so that you can immediately remove the lowest probability prefix and replace it with a better one
        2) Run the neural network to get the next word probabilities for the whole beam at once rather than for one prefix at a time
        3) Further to point 2, store the prefixes in a matrix with a number of columns equal to the maximum length (use padding) rather than using lists to which you append more indexes
        4) You can also add more features such as ignoring probabilities that are zero and not repeating the same word twice
        '''
        
        fig.show()