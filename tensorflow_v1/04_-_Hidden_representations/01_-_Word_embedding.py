import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.02
momentum = 0.5
max_epochs = 3000
init_stddev = 0.0001

#The training set of sentences
tokens = [
        '<BEG> the dog barks and the cat meows <END>'.split(' '),
        '<BEG> the cat meows and the dog barks <END>'.split(' '),
    ]

#Turn the above sentences into token trigrams
token_trigrams = [ tuple(sent[i:i+3]) for sent in tokens for i in range(len(sent)-2) ]

#Extract vocabulary of unique words
vocab = sorted({ token for sent in tokens for token in sent })

#Replace all words in the above sentences with indexes (numbers) according to their position in the vocabulary
token2index = { token: index for (index, token) in enumerate(vocab) }
indexes = [ [ token2index[token] for token in sent ] for sent in tokens ] #Sentences with indexed words
trigrams = np.array([ sent[i:i+3] for sent in indexes for i in range(len(sent)-2) ], np.int32) #Matrix of trigrams consisting of indexed words

g = tf.Graph()
with g.as_default():
    lefts   = tf.placeholder(tf.int32, [None], 'lefts') #Left word in trigram
    rights  = tf.placeholder(tf.int32, [None], 'rights') #Right word in trigram
    targets = tf.placeholder(tf.int32, [None], 'targets') #Middle word in trigram

    #An embedding matrix is a matrix with a row vector for each unique word (gets optimised with the rest of the neural network)
    embedding_matrix = tf.get_variable('embedding_matrix', [ len(vocab), 2 ], tf.float32, tf.random_normal_initializer(stddev=init_stddev))

    #The context of the middle word is the left and right words, embedded and concatenated into a single vector
    embedded_lefts  = tf.nn.embedding_lookup(embedding_matrix, lefts)
    embedded_rights = tf.nn.embedding_lookup(embedding_matrix, rights)
    embedded_context = tf.concat([ embedded_lefts, embedded_rights ], axis=1)

    #Predict the middle word from the context
    W = tf.get_variable('W', [2*2, len(vocab)], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    b = tf.get_variable('b', [len(vocab)], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(embedded_context, W) + b
    probs = tf.nn.softmax(logits)
    
    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    
    step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error', sep='\t')
        for epoch in range(1, max_epochs+1):
            s.run([ step ], { lefts: trigrams[:, 0], rights: trigrams[:, 2], targets: trigrams[:, 1] })

            [ train_error ] = s.run([ error ], { lefts: trigrams[:, 0], rights: trigrams[:, 2], targets: trigrams[:, 1] })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error, sep='\t')
                
                [ curr_embeddings ] = s.run([ embedding_matrix ], { })

                ax[0].cla()
                for (token, vector) in zip(vocab, curr_embeddings.tolist()):
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
                ax[1].set_xlim(-10, max_epochs)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0.0, 2.0)
                ax[1].set_ylabel('XE') #Cross entropy
                ax[1].grid(True)
                ax[1].set_title('Error progress')
                ax[1].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()
        print('token', 'vector', sep='\t')
        for (token, vector) in zip(vocab, curr_embeddings.tolist()):
            print(token, np.round(vector, 3), sep='\t')
        print()
        
        [ curr_probs ] = s.run([ probs ], { lefts: trigrams[:, 0], rights: trigrams[:, 2] })
        trigrams_shown = set()
        print('3gram', 'top 3 predicted middle tokens', sep='\t')
        for (trigram, ps) in zip(token_trigrams, curr_probs):
            if trigram not in trigrams_shown: 
                top_probs = sorted(zip(ps, vocab), reverse=True)[:3]
                print(' '.join(trigram), ' '.join([ '{} ({:.5f})'.format(t, p) for (p, t) in top_probs ]), sep='\t')
                trigrams_shown.add(trigram)
                
        fig.show()