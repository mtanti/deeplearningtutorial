import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

token_sents = [
        '<BEG> the dog barks and the cat meows <END>'.split(' '),
        '<BEG> the cat meows and the dog barks <END>'.split(' '),
    ]

token_trigrams = [ tuple(sent[i:i+3]) for sent in token_sents for i in range(len(sent)-2) ]

vocab = sorted({ token for sent in token_sents for token in sent })

token2index = { token: index for (index, token) in enumerate(vocab) }
index_sents = [ [ token2index[token] for token in sent ] for sent in token_sents ]
index_trigrams = np.array([ sent[i:i+3] for sent in index_sents for i in range(len(sent)-2) ], np.int32)
index_lefts = index_trigrams[:,0]
index_rights = index_trigrams[:,2]
index_targets = index_trigrams[:,1]

###################################

class Model(object):

    def __init__(self, vocab_size):
        learning_rate = 0.1
        momentum = 0.9
        init_stddev = 1e-2
        embed_size = 2
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.lefts   = tf.placeholder(tf.int32, [None], 'lefts') #Left word in trigram as index.
            self.rights  = tf.placeholder(tf.int32, [None], 'rights') #Right word in trigram as index.
            self.targets = tf.placeholder(tf.int32, [None], 'targets') #Middle word in trigram as index.

            self.params = []
            
            with tf.variable_scope('embeddings'):
                #An embedding matrix is a matrix with a row vector for each unique word (gets optimised with the rest of the neural network).
                self.embedding_matrix = tf.get_variable('embedding_matrix', [ vocab_size, embed_size ], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.params.extend([ embedding_matrix ])

                embedded_lefts  = tf.nn.embedding_lookup(self.embedding_matrix, self.lefts)
                embedded_rights = tf.nn.embedding_lookup(self.embedding_matrix, self.rights)
                embedded_context = tf.concat([ embedded_lefts, embedded_rights ], axis=1)

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [2*embed_size, vocab_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [vocab_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                logits = tf.matmul(embedded_context, W) + b
                self.probs = tf.nn.softmax(logits)
            
            self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits))
            
            self.optimiser_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.error)
        
            self.init = tf.global_variables_initializer()
            
            self.graph.finalize()

            self.sess = tf.Session()
    
    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, lefts, rights, targets):
        return self.sess.run([ self.optimiser_step ], { self.lefts: lefts, self.rights: rights, self.targets: targets })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_error(self, lefts, rights, targets):
        return self.sess.run([ self.error ], { self.lefts: lefts, self.rights: rights, self.targets: targets })[0]
    
    def predict(self, lefts, rights):
        return self.sess.run([ self.probs ], { self.lefts: lefts, self.rights: rights })[0]
    
    def get_embeddings(self):
        return self.sess.run([ self.embedding_matrix ], { })[0]

###################################

max_epochs = 1000

(fig, axs) = plt.subplots(1, 2)

embedding_plots = list()
embedding_texts = list()
for token in vocab:
    [ embedding_plot ] = axs[0].plot([ 0 ], [ 0 ], linestyle='', marker='o', markersize=10)
    embedding_plots.append(embedding_plot)
    embedding_text = axs[0].text(0, 0, token, fontdict={ 'fontsize': 8 })
    embedding_texts.append(embedding_text)
axs[0].set_xlim(-5.0, 5.0)
axs[0].set_xlabel('d0')
axs[0].set_ylim(-5.0, 5.0)
axs[0].set_ylabel('d1')
axs[0].grid(True)
axs[0].set_title('Embeddings')

[ train_error_plot ] = axs[1].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[1].set_xlim(0, max_epochs)
axs[1].set_xlabel('epoch')
axs[1].set_ylim(0.0, 2.0)
axs[1].set_ylabel('XE')
axs[1].grid(True)
axs[1].set_title('Error progress')
axs[1].legend()

fig.tight_layout()
fig.show()

###################################

model = Model(len(vocab))
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(index_lefts, index_rights, index_targets)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        
        embeddings = model.get_embeddings()
        
        for (embedding_plot, embedding_text, embedding) in zip(embedding_plots, embedding_texts, embeddings.tolist()):
            embedding_plot.set_data([ embedding[0] ], [ embedding[1] ])
            embedding_text.set_position( (embedding[0], embedding[1]) )
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(index_lefts, index_rights, index_targets)

print()
print('token', 'vector', sep='\t')
embeddings = model.get_embeddings()
for (token, embedding) in zip(vocab, embeddings.tolist()):
    print(token, np.round(embedding, 3), sep='\t')
print()

probs = model.predict(index_lefts, index_rights)
trigrams_shown = set()
print('3gram', 'top 3 predicted middle tokens', sep='\t')
for (trigram, ps) in zip(token_trigrams, probs):
    if trigram not in trigrams_shown: 
        top_probs = sorted(zip(ps, vocab), reverse=True)[:3]
        print(' '.join(trigram), ' '.join([ '{} ({:.5f})'.format(t, p) for (p, t) in top_probs ]), sep='\t')
        trigrams_shown.add(trigram)

model.close()