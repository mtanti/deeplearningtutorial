import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 1.0
max_epochs = 2000
init_stddev = 0.01

tokens = [
        '<BEG> the dog barks and the cat meows <END>'.split(' '),
        '<BEG> the cat meows and the dog barks <END>'.split(' '),
    ]
tags = [
        'EDGE DET NOUN VERB CONJ DET NOUN VERB EDGE'.split(' '),
        'EDGE DET NOUN VERB CONJ DET NOUN VERB EDGE'.split(' '),
    ]

#Prepare dataset for task 1
token_trigrams = [ tuple(sent[i:i+3]) for sent in tokens for i in range(len(sent)-2) ]
token_vocab = sorted({ token for sent in tokens for token in sent })
token2index = { token: index for (index, token) in enumerate(token_vocab) }
token_indexes = [ [ token2index[token] for token in sent ] for sent in tokens ]
token_index_trigrams = np.array([ sent[i:i+3] for sent in token_indexes for i in range(len(sent)-2) ], np.int32)

#Prepare dataset for task 2
tag_trigrams = [ tag for sent in tags for tag in sent[1:-1] ]
tag_vocab = sorted({ tag for sent in tags for tag in sent })
tag2index = { tag: index for (index, tag) in enumerate(tag_vocab) }
tag_indexes = [ [ tag2index[tag] for tag in sent ] for sent in tags ]
tag_index_trigrams = [ tag for sent in tag_indexes for tag in sent[1:-1] ]

g = tf.Graph()
with g.as_default():
    #Shared parameters
    embedding_matrix = tf.get_variable('embedding_matrix', [len(token_vocab), 2], tf.float32, tf.random_normal_initializer(stddev=init_stddev))

    #Module for task 1
    with tf.variable_scope('word_prediction'):
        lefts = tf.placeholder(tf.int32, [None], 'lefts')
        rights = tf.placeholder(tf.int32, [None], 'rights')
        pred_targets = tf.placeholder(tf.int32, [None], 'pred_targets')

        embedded_lefts = tf.nn.embedding_lookup(embedding_matrix, lefts)
        embedded_rights = tf.nn.embedding_lookup(embedding_matrix, rights)
        embedded_context = tf.concat([ embedded_lefts, embedded_rights ], axis=1)

        W = tf.get_variable('W', [4, len(token_vocab)], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        b = tf.get_variable('b', [len(token_vocab)], tf.float32, tf.zeros_initializer())
        pred_logits = tf.matmul(embedded_context, W) + b
        pred_probs = tf.nn.softmax(pred_logits)

    #Module for task 2
    with tf.variable_scope('tagger'):
        phrases = tf.placeholder(tf.int32, [None, 3], 'phrases')
        tag_targets = tf.placeholder(tf.int32, [None], 'tag_targets')

        embedded_phrases = tf.reshape(tf.nn.embedding_lookup(embedding_matrix, phrases), [-1, 2*3]) #Reusing the same embedding matrix variable defined above (this is called parameter sharing)
        
        W = tf.get_variable('W', [2*3, len(tag_vocab)], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        b = tf.get_variable('b', [len(tag_vocab)], tf.float32, tf.zeros_initializer())
        tag_logits = tf.matmul(embedded_phrases, W) + b
        tag_probs = tf.nn.softmax(tag_logits)

    #Train both outputs/tasks at once (multi-objective optimisation, just like weight decay regularisation)
    error = (
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pred_targets, logits=pred_logits))
            +
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tag_targets, logits=tag_logits))
        )
    
    step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error', sep='\t')
        for epoch in range(1, max_epochs+1):
            s.run([ step ], { lefts: token_index_trigrams[:, 0], rights: token_index_trigrams[:, 2], phrases: token_index_trigrams, pred_targets: token_index_trigrams[:, 1], tag_targets: tag_index_trigrams })

            [ train_error ] = s.run([ error ], { lefts: token_index_trigrams[:, 0], rights: token_index_trigrams[:, 2], phrases: token_index_trigrams, pred_targets: token_index_trigrams[:, 1], tag_targets: tag_index_trigrams  })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error, sep='\t')
                
                [ curr_embeddings ] = s.run([ embedding_matrix ], { })

                ax[0].cla()
                for (token, vector) in zip(token_vocab, curr_embeddings.tolist()):
                    ax[0].plot(vector[0], vector[1], linestyle='', marker='o', markersize=10)
                    ax[0].text(vector[0], vector[1], token)
                ax[0].set_xlim(-5.0, 5.0)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(-5.0, 5.0)
                ax[0].set_ylabel('x1')
                ax[0].grid(True)
                ax[0].set_title('Embeddings')

                ax[1].cla()
                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].set_xlim(0, max_epochs)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0.0, 0.1)
                ax[1].set_ylabel('XE') #Cross entropy
                ax[1].grid(True)
                ax[1].set_title('Error progress')
                ax[1].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()
        [ curr_probs ] = s.run([ pred_probs ], { lefts: token_index_trigrams[:, 0], rights: token_index_trigrams[:, 2] })
        trigrams_shown = set()
        print('3gram', 'top 3 middle word predictions', sep='\t')
        for (trigram, ps) in zip(token_trigrams, curr_probs):
            if trigram not in trigrams_shown: 
                top_probs = sorted(zip(ps, token_vocab), reverse=True)[:3]
                print(' '.join(trigram), ' '.join([ '{} ({:.5f})'.format(t, p) for (p, t) in top_probs ]), sep='\t')
                trigrams_shown.add(trigram)

        print()
        print('3gram', 'top 3 middle word POS tags`', sep='\t')
        [ curr_probs ] = s.run([ tag_probs ], { phrases: token_index_trigrams })
        trigrams_shown = set()
        for (trigram, ps) in zip(token_trigrams, curr_probs):
            if trigram not in trigrams_shown: 
                top_probs = sorted(zip(ps, tag_vocab), reverse=True)[:3]
                print(' '.join(trigram), ' '.join([ '{} ({:.5f})'.format(t, p) for (p, t) in top_probs ]), sep='\t')
                trigrams_shown.add(trigram)
                
        fig.show()