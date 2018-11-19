import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

tokens = [
        '<BEG> the dog barks and the cat meows <END>'.split(' '),
        '<BEG> the cat meows and the dog barks <END>'.split(' '),
    ]
tags = [
        'EDGE DET NOUN VERB CONJ DET NOUN VERB EDGE'.split(' '),
        'EDGE DET NOUN VERB CONJ DET NOUN VERB EDGE'.split(' '),
    ]

print('training embeddings')

token_trigrams = [ tuple(sent[i:i+3]) for sent in tokens for i in range(len(sent)-2) ]
token_vocab = sorted({ token for sent in tokens for token in sent })
token2index = { token: index for (index, token) in enumerate(token_vocab) }
token_indexes = [ [ token2index[token] for token in sent ] for sent in tokens ]
token_index_trigrams = np.array([ sent[i:i+3] for sent in token_indexes for i in range(len(sent)-2) ], np.int32)

g = tf.Graph()
with g.as_default():
    lefts   = tf.placeholder(tf.int32, [None], 'lefts')
    rights  = tf.placeholder(tf.int32, [None], 'rights')
    targets = tf.placeholder(tf.int32, [None], 'targets')

    embedding_matrix = tf.get_variable('embedding_matrix', [ len(token_vocab), 2 ], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))

    embedded_lefts  = tf.nn.embedding_lookup(embedding_matrix, lefts)
    embedded_rights = tf.nn.embedding_lookup(embedding_matrix, rights)
    embedded_context = tf.concat([ embedded_lefts, embedded_rights ], axis=1)

    W = tf.get_variable('W', [2*2, len(token_vocab)], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    b = tf.get_variable('b', [len(token_vocab)], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(embedded_context, W) + b
    probs = tf.nn.softmax(logits)
    
    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    
    step = tf.train.GradientDescentOptimizer(1.0).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        for epoch in range(1, 2000+1):
            s.run([ step ], { lefts: token_index_trigrams[:, 0], rights: token_index_trigrams[:, 2], targets: token_index_trigrams[:, 1] })

        #Get the embedding matrix values
        [ curr_embeddings ] = s.run([ embedding_matrix ], { })

print('finished training embeddings')

print('training tagger')

tag_trigrams = [ tag for sent in tags for tag in sent[1:-1] ]
tag_vocab = sorted({ tag for sent in tags for tag in sent })
tag2index = { tag: index for (index, tag) in enumerate(tag_vocab) }
tag_indexes = [ [ tag2index[tag] for tag in sent ] for sent in tags ]
tag_index_trigrams = [ tag for sent in tag_indexes for tag in sent[1:-1] ]

g = tf.Graph()
with g.as_default():
    phrases = tf.placeholder(tf.int32, [None, 3], 'phrases')
    targets = tf.placeholder(tf.int32, [None], 'targets')

    #This is where transfer learning happens (use a variable instead of a constant to perform fine-tuned transfer learning)
    embedding_matrix = tf.constant(curr_embeddings, tf.float32, [ len(token_vocab), 2 ], 'embedding_matrix')

    embedded_phrases = tf.reshape(tf.nn.embedding_lookup(embedding_matrix, phrases), [-1, 2*3])

    W = tf.get_variable('W', [2*3, len(tag_vocab)], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    b = tf.get_variable('b', [len(tag_vocab)], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(embedded_phrases, W) + b
    probs = tf.nn.softmax(logits)
    
    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    
    step = tf.train.GradientDescentOptimizer(1.0).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 1)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 200+1):
            s.run([ step ], { phrases: token_index_trigrams, targets: tag_index_trigrams })

            [ train_error ] = s.run([ error ], { phrases: token_index_trigrams, targets: tag_index_trigrams })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error)
                
                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax.set_xlim(0, 200)
                ax.set_xlabel('epoch')
                ax.set_ylim(0.0, 0.1)
                ax.set_ylabel('XE') #Cross entropy
                ax.grid(True)
                ax.set_title('Error progress')
                ax.legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()
        [ curr_probs ] = s.run([ probs ], { phrases: token_index_trigrams })
        trigrams_shown = set()
        for (trigram, ps) in zip(token_trigrams, curr_probs):
            if trigram not in trigrams_shown: 
                top_probs = sorted(zip(ps, tag_vocab), reverse=True)[:3]
                print(trigram, top_probs)
                trigrams_shown.add(trigram)
        fig.show()
        
