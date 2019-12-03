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

###################################

class Model(object):

    def __init__(self, vocab_size):
        learning_rate = 1.0
        momentum = 0.9
        init_stddev = 1e-2
        embed_size = 2
        kernel_width = 2
        kernel_size = 2
        sent_size = 4 #We now have to specify the size of sentence we expect.
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sents   = tf.placeholder(tf.int32, [None, sent_size], 'sents')
            self.targets = tf.placeholder(tf.float32, [None, 1], 'targets')

            self.params = []

            batch_size = tf.shape(self.sents)[0] #Get the number of sentences in the input.
            
            with tf.variable_scope('embeddings'):
                self.embedding_matrix = tf.get_variable('embedding_matrix', [ vocab_size, embed_size ], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.params.extend([ self.embedding_matrix ])

                embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.sents)

            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', [kernel_width, embed_size, kernel_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [kernel_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.conv_hs = tf.sigmoid(tf.nn.conv1d(embedded, W, 1, 'VALID') + b)

                #The number of convolutions or 'slides' the kernel will make over the sentence need to be known prior to running the graph because they need to be used to set the weight matrix size in the output layer.
                num_convs = sent_size - kernel_width + 1
                vec_size_per_sent = num_convs*kernel_size
                self.flat_hs = tf.reshape(self.conv_hs, [ batch_size, vec_size_per_sent ])

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [vec_size_per_sent, 1], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
                logits = tf.matmul(self.flat_hs, W) + b
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

###################################

max_epochs = 1000

(fig, ax) = plt.subplots(1, 1)

[ train_error_plot ] = ax.plot([], [], color='red', linestyle='-', linewidth=1, label='train')
ax.set_xlim(0, max_epochs)
ax.set_xlabel('epoch')
ax.set_ylim(0.0, 2.0)
ax.set_ylabel('XE')
ax.grid(True)
ax.set_title('Error progress')
ax.legend()

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
        
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(index_sents, sentiments)

print()
probs = model.predict(index_sents)
print('sent', 'sentiment', sep='\t')
for (token_sent, prob) in zip(token_sents, probs.tolist()):
    print(' '.join(token_sent), np.round(prob[0], 3), sep='\t')
    
model.close()