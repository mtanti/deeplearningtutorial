import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

sent_len = 4

tokens = [
        'i like it <PAD>'.split(' '),  #positive
        'i hate it <PAD>'.split(' '),  #negative
        'i don\'t hate it'.split(' '), #positive
        'i don\'t like it'.split(' '), #negative
    ]
sentiments = [
        [1],
        [0],
        [1],
        [0]
    ]

vocab = sorted({ token for sent in tokens for token in sent })
token2index = { token: index for (index, token) in enumerate(vocab) }
indexes = [ [ token2index[token] for token in sent ] for sent in tokens ]

token_bigrams = sorted({ tuple(sent[i:i+2]) for sent in tokens for i in range(len(sent)-1) })
bigrams = np.array([ [ token2index[sent[i]], token2index[sent[i+1]] ] for sent in token_bigrams for i in range(len(sent)-1) ], np.int32)

g = tf.Graph()
with g.as_default():
    sents = tf.placeholder(tf.int32, [None, None], 'sents')
    targets = tf.placeholder(tf.float32, [None, 1], 'targets')

    embedding_size = 2
    embedding_matrix = tf.get_variable('embedding_matrix', [ len(vocab), embedding_size ], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    embedded = tf.nn.embedding_lookup(embedding_matrix, sents)

    window_width = 2
    hidden_layer_size = 2
    W = tf.get_variable('W', [window_width, embedding_size, hidden_layer_size], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    b = tf.get_variable('b', [hidden_layer_size], tf.float32, tf.zeros_initializer())
    conv_hs = tf.sigmoid(tf.nn.conv1d(embedded, W, 1, 'VALID') + b)

    num_windows_per_sent = sent_len - window_width + 1
    
    hs = tf.reduce_max(conv_hs, axis=1)
    
    W2 = tf.get_variable('W2', [hidden_layer_size, 1], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    b2 = tf.get_variable('b2', [1], tf.float32)
    logits = tf.matmul(hs, W2) + b2
    probs = tf.sigmoid(logits)

    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    
    step = tf.train.GradientDescentOptimizer(1.0).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 5000+1):
            s.run([ step ], { sents: indexes, targets: sentiments })

            [ train_error ] = s.run([ error ], { sents: indexes, targets: sentiments })
            train_errors.append(train_error)
            
            if epoch%100 == 0:
                print(epoch, train_error)
                
                [ curr_bigram_vecs ] = s.run([ conv_hs ], { sents: bigrams })

                ax[0].cla()
                for (token_bigram, vector) in zip(token_bigrams, curr_bigram_vecs[:,0,:].tolist()):
                    ax[0].plot(vector[0], vector[1], linestyle='', marker='o', markersize=10)
                    ax[0].text(vector[0], vector[1], ' '.join(token_bigram))
                ax[0].set_xlim(0, 1)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(0, 1)
                ax[0].set_ylabel('x1')
                ax[0].grid(True)
                ax[0].set_title('Bigrams')

                ax[1].cla()
                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].set_xlim(0, 5000)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0.0, 1.0)
                ax[1].set_ylabel('XE') #Cross entropy
                ax[1].grid(True)
                ax[1].set_title('Error progress')
                ax[1].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()
        
        [ curr_probs ] = s.run([ probs ], { sents: indexes })
        for (sent, prob) in zip(tokens, curr_probs[:,0].tolist()):
            print(' '.join(sent), round(prob, 2))
        fig.show()
