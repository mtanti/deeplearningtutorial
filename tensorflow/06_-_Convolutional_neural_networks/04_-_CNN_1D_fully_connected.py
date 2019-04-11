import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

max_epochs = 4000
init_stddev = 0.01
sent_len = 4 #The number of tokens in each sentence (pad words are used to make shorter sentences longer)
window_width = 2
hidden_layer_size = 2
embedding_size = 2

tokens = [
        'i like it <PAD>'.split(' '),  #positive
        'i hate it <PAD>'.split(' '),  #negative
        'i don\'t hate it'.split(' '), #positive
        'i don\'t like it'.split(' '), #negative
    ]
sentiments = [
        [ 1 ],
        [ 0 ],
        [ 1 ],
        [ 0 ]
    ]

vocab = sorted({ token for sent in tokens for token in sent })
token2index = { token: index for (index, token) in enumerate(vocab) }
indexes = [ [ token2index[token] for token in sent ] for sent in tokens ]

token_bigrams = sorted({ tuple(sent[i:i+2]) for sent in tokens for i in range(len(sent)-1) })
bigrams = np.array([ [ token2index[sent[i]], token2index[sent[i+1]] ] for sent in token_bigrams for i in range(len(sent)-1) ], np.int32)

g = tf.Graph()
with g.as_default():
    sents = tf.placeholder(tf.int32, [None, sent_len], 'sents')
    targets = tf.placeholder(tf.float32, [None, 1], 'targets')
    
    #The amount of sentences given at once
    batch_size = tf.shape(sents)[0]

    embedding_matrix = tf.get_variable('embedding_matrix', [len(vocab), embedding_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    embedded = tf.nn.embedding_lookup(embedding_matrix, sents)

    W = tf.get_variable('W', [window_width, embedding_size, hidden_layer_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    b = tf.get_variable('b', [hidden_layer_size], tf.float32, tf.zeros_initializer())
    conv_hs = tf.sigmoid(tf.nn.conv1d(embedded, W, 1, 'VALID') + b)

    #Reshape the window vectors of each image so that they are one flat vector per image
    num_windows_per_sent = sent_len - window_width + 1
    vec_size_per_sent = num_windows_per_sent*hidden_layer_size
    hs = tf.reshape(conv_hs, [batch_size, vec_size_per_sent])
    
    W2 = tf.get_variable('W2', [vec_size_per_sent, 1], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    b2 = tf.get_variable('b2', [1], tf.float32)
    logits = tf.matmul(hs, W2) + b2
    probs = tf.sigmoid(logits)

    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    
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
            s.run([ step ], { sents: indexes, targets: sentiments })

            [ train_error ] = s.run([ error ], { sents: indexes, targets: sentiments })
            train_errors.append(train_error)
            
            if epoch%100 == 0:
                print(epoch, train_error, sep='\t')
                
                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax.set_xlim(0, max_epochs)
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
        
        [ sent_vecs ] = s.run([ hs ], { sents: indexes })
        print('sentence', 'vector', sep='\t')
        for (token_sent, vector) in zip(tokens, sent_vecs.tolist()):
            print(' '.join(token_sent), np.round(vector, 3), sep='\t')
        print()

        [ curr_probs ] = s.run([ probs ], { sents: indexes })
        print('sentence', 'sentiment', sep='\t')
        for (sent, prob) in zip(tokens, curr_probs[:,0].tolist()):
            print(' '.join(sent), round(prob, 2), sep='\t')
            
        fig.show()