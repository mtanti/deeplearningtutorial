import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

token_sents = [
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

vocab = sorted({ token for sent in token_sents for token in sent })

token2index = { token: index for (index, token) in enumerate(vocab) }
index_sents = [ [ token2index[token] for token in sent ] for sent in token_sents ]

token_bigrams = sorted({ tuple(sent[i:i+2]) for sent in token_sents for i in range(len(token_sents)-1) })
index_bigrams = [ [ token2index[token] for token in token_bigram ] for token_bigram in token_bigrams ]

###################################

class Model(object):

    def __init__(self, vocab_size):
        learning_rate = 1.0
        momentum = 0.9
        init_stddev = 1e-2
        embed_size = 2
        kernel_width = 2 #Number of words to see every 'slide'.
        kernel_size = 2 #Vector size after convolving.
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sents   = tf.placeholder(tf.int32, [None, None], 'sents')
            self.targets = tf.placeholder(tf.float32, [None, 1], 'targets')

            self.params = []
            
            with tf.variable_scope('embeddings'):
                self.embedding_matrix = tf.get_variable('embedding_matrix', [ vocab_size, embed_size ], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.params.extend([ self.embedding_matrix ])

                embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.sents)

            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', [kernel_width, embed_size, kernel_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [kernel_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.conv_hs = tf.sigmoid(tf.nn.conv1d(embedded, W, 1, 'VALID') + b)

                self.pool_hs = tf.reduce_max(self.conv_hs, axis=1) #Max pooling

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [kernel_size, 1], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
                logits = tf.matmul(self.pool_hs, W) + b
                self.probs = tf.sigmoid(logits)
            
            self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=logits))
            
            self.optimiser_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.error)
        
            self.init = tf.global_variables_initializer()
            
            self.graph.finalize()

            self.sess = tf.Session()
    
    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, sents, targets):
        return self.sess.run([ self.optimiser_step ], { self.sents: sents, self.targets: targets })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_error(self, sents, targets):
        return self.sess.run([ self.error ], { self.sents: sents, self.targets: targets })[0]
    
    def predict(self, sents):
        return self.sess.run([ self.probs ], { self.sents: sents })[0]
    
    def get_conv(self, sents):
        return self.sess.run([ self.conv_hs ], { self.sents: sents })[0]

    def get_pool(self, sents):
        return self.sess.run([ self.pool_hs ], { self.sents: sents })[0]

###################################

max_epochs = 1000

(fig, axs) = plt.subplots(1, 3)

bigram_plots = list()
bigram_texts = list()
for token_bigram in token_bigrams:
    [ bigram_plot ] = axs[0].plot([ 0 ], [ 0 ], linestyle='', marker='o', markersize=10)
    bigram_plots.append(bigram_plot)
    bigram_text = axs[0].text(0, 0, ' '.join(token_bigram), fontdict={ 'fontsize': 8 })
    bigram_texts.append(bigram_text)
axs[0].set_xlim(0.0, 1.0)
axs[0].set_xlabel('d0')
axs[0].set_ylim(0.0, 1.0)
axs[0].set_ylabel('d1')
axs[0].grid(True)
axs[0].set_title('Bigrams')

sent_plots = list()
sent_texts = list()
for token_sent in token_sents:
    [ sent_plot ] = axs[1].plot([ 0 ], [ 0 ], linestyle='', marker='o', markersize=10)
    sent_plots.append(sent_plot)
    sent_text = axs[1].text(0, 0, ' '.join(token_sent), fontdict={ 'fontsize': 8 })
    sent_texts.append(sent_text)
axs[1].set_xlim(0.0, 1.0)
axs[1].set_xlabel('d0')
axs[1].set_ylim(0.0, 1.0)
axs[1].set_ylabel('d1')
axs[1].grid(True)
axs[1].set_title('Sents')

[ train_error_plot ] = axs[2].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[2].set_xlim(0, max_epochs)
axs[2].set_xlabel('epoch')
axs[2].set_ylim(0.0, 2.0)
axs[2].set_ylabel('XE')
axs[2].grid(True)
axs[2].set_title('Error progress')
axs[2].legend()

fig.tight_layout()
fig.show()

###################################

model = Model(len(vocab))
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(index_sents, sentiments)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        
        convs = model.get_conv(index_bigrams)
        pools = model.get_pool(index_sents)
        
        for (bigram_plot, bigram_text, conv) in zip(bigram_plots, bigram_texts, convs.tolist()):
            bigram_plot.set_data([ conv[0][0] ], [ conv[0][1] ])
            bigram_text.set_position( (conv[0][0], conv[0][1]) )
        for (sent_plot, sent_text, pool) in zip(sent_plots, sent_texts, pools.tolist()):
            sent_plot.set_data([ pool[0] ], [ pool[1] ])
            sent_text.set_position( (pool[0], pool[1]) )
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(index_sents, sentiments)

print()
print('bigram', 'vector', sep='\t')
convs = model.get_conv(index_bigrams)
for (token_bigram, conv) in zip(token_bigrams, convs.tolist()):
    print(' '.join(token_bigram), np.round(conv[0], 3), sep='\t')
print()
print('sent', 'vector', sep='\t')
pools = model.get_pool(index_sents)
for (token_sent, pool) in zip(token_sents, pools.tolist()):
    print(' '.join(token_sent), np.round(pool, 3), sep='\t')
print()

probs = model.predict(index_sents)
print('sent', 'sentiment', sep='\t')
for (token_sent, prob) in zip(token_sents, probs.tolist()):
    print(' '.join(token_sent), np.round(prob[0], 3), sep='\t')
    
model.close()