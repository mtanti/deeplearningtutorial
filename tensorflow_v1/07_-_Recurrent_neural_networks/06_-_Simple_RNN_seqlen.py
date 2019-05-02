import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

max_epochs = 5000
init_stddev = 0.001
embedding_size = 2
state_size = 2

tokens = [
        'i like it'.split(' '),  #positive
        'i hate it'.split(' '),  #negative
        'i don\'t hate it'.split(' '), #positive
        'i don\'t like it'.split(' '), #negative
    ]
token_lens = [
        3,
        3,
        4,
        4,
    ] #Alternatively, you can replace this list with [ len(sent) for sent in tokens ]
sentiments = [
        [ 1 ],
        [ 0 ],
        [ 1 ],
        [ 0 ],
    ]

max_len = max(token_lens)
vocab = sorted({ token for sent in tokens for token in sent })
token2index = { token: index for (index, token) in enumerate(vocab) }
indexes = [ [ token2index[token] for token in sent ] + [ 0 for _ in range(max_len - len(sent)) ] for sent in tokens ] #Add zeros to the end of each sentence so that all sentences are equal to the maximum length


class Cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self):
        super().__init__()
        self.W = None
        self.b = None

    @property
    def state_size(self):
        return state_size

    @property
    def output_size(self):
        return state_size

    def build(self, inputs_shape):
        self.W = self.add_variable('W', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=0.5)) #RNNs might need more varied initial weights to work well
        self.b = self.add_variable('b', [state_size], tf.float32, tf.zeros_initializer())
        
    def call(self, x, curr_state):
        state_input = tf.concat([ curr_state, x ], axis=1)
        new_state = tf.tanh(tf.matmul(state_input, self.W) + self.b)
        return (new_state, new_state)


g = tf.Graph()
with g.as_default():
    sents = tf.placeholder(tf.int32, [None, None], 'sents')
    sent_lens = tf.placeholder(tf.int32, [None], 'sent_lens')
    targets = tf.placeholder(tf.float32, [None, 1], 'targets')
    
    batch_size = tf.shape(sents)[0]
    
    embedding_matrix = tf.get_variable('embedding_matrix', [len(vocab), embedding_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    embedded = tf.nn.embedding_lookup(embedding_matrix, sents)
    
    init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    batch_init = tf.tile(tf.reshape(init_state, [1, state_size]), [batch_size, 1])
    
    cell = Cell()
    (_, state) = tf.nn.dynamic_rnn(cell, embedded, sequence_length=sent_lens, initial_state=batch_init)
    
    W = tf.get_variable('W', [state_size, 1], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(state, W) + b
    probs = tf.sigmoid(logits)

    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    
    step = tf.train.AdamOptimizer().minimize(error)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 3)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error', sep='\t')
        for epoch in range(1, max_epochs+1):
            s.run([ step ], { sents: indexes, sent_lens: token_lens, targets: sentiments })

            [ train_error ] = s.run([ error ], { sents: indexes, sent_lens: token_lens, targets: sentiments })
            train_errors.append(train_error)
            
            if epoch%100 == 0:
                print(epoch, train_error, sep='\t')

                [ curr_embeddings ] = s.run([ embedding_matrix ], { })
                ax[0].cla()
                for (token, token_vec) in zip(vocab, curr_embeddings.tolist()):
                    ax[0].plot(token_vec[0], token_vec[1], linestyle='', marker='o', markersize=10)
                    ax[0].text(token_vec[0], token_vec[1], token)
                ax[0].set_xlim(-2, 2)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(-2, 2)
                ax[0].set_ylabel('x1')
                ax[0].grid(True)
                ax[0].set_title('Embeddings')
                
                [ curr_state ] = s.run([ state ], { sents: indexes, sent_lens: token_lens })
                ax[1].cla()
                for (sent, vec) in zip(tokens, curr_state.tolist()):
                    ax[1].plot(vec[0], vec[1], linestyle='', marker='o', markersize=10)
                    ax[1].text(vec[0], vec[1], ' '.join(sent))
                ax[1].set_xlim(-1, 1)
                ax[1].set_xlabel('x0')
                ax[1].set_ylim(-1, 1)
                ax[1].set_ylabel('x1')
                ax[1].grid(True)
                ax[1].set_title('Sentences')
                
                ax[2].cla()
                ax[2].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[2].set_xlim(0, max_epochs)
                ax[2].set_xlabel('epoch')
                ax[2].set_ylim(0.0, 1.0)
                ax[2].set_ylabel('XE') #Cross entropy
                ax[2].grid(True)
                ax[2].set_title('Error progress')
                ax[2].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()
        
        [ curr_state ] = s.run([ state ], { sents: indexes, sent_lens: token_lens })
        print('sentence', 'vector', sep='\t')
        for (sent, vec) in zip(tokens, curr_state.tolist()):
            print(' '.join(sent), np.round(vec, 2), sep='\t')
        print()
        
        [ curr_probs ] = s.run([ probs ], { sents: indexes, sent_lens: token_lens })
        print('sentence', 'sentiment', sep='\t')
        for (sent, prob) in zip(tokens, curr_probs[:, 0].tolist()):
            print(' '.join(sent), round(prob, 2), sep='\t')
        print()
        
        fig.show()