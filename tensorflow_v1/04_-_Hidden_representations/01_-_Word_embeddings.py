import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

#The training set of sentences.
token_sents = [
        '<BEG> the dog barks and the cat meows <END>'.split(' '),
        '<BEG> the cat meows and the dog barks <END>'.split(' '),
    ]

#Turn the above sentences into token trigrams.
token_trigrams = [ tuple(sent[i:i+3]) for sent in token_sents for i in range(len(sent)-2) ]

#Extract vocabulary of unique words.
vocab = sorted({ token for sent in token_sents for token in sent })
vocab_size = len(vocab)

#Replace all words in the above sentences with indexes (numbers) according to their position in the vocabulary.
token2index = { token: index for (index, token) in enumerate(vocab) }
index_sents = [ [ token2index[token] for token in sent ] for sent in token_sents ] #Sentences with indexed words.
index_trigrams = np.array([ sent[i:i+3] for sent in index_sents for i in range(len(sent)-2) ], np.int32) #Matrix of trigrams consisting of indexed words.
index_lefts = index_trigrams[:,0] #The left side of the trigram.
index_rights = index_trigrams[:,2] #The right side of the trigram.
index_targets = index_trigrams[:,1] #The middle of the trigram.

#Replace all indexes with their corresponding one-hot vectors to transform each word into a vector that is equally different from every other word.
index2onehot = np.eye(vocab_size).tolist() #Each row of an identity matrix of size equal to the vocabulary size is a one-hot vector of a word in the vocabulary.
token2onehot = { token: index2onehot[token2index[token]] for token in vocab }
onehot_sents = [ [ index2onehot[index] for index in sent ] for sent in index_sents ] #Sentences with one-hot encoded words.
onehot_trigrams = np.array([ sent[i:i+3] for sent in onehot_sents for i in range(len(sent)-2) ], np.float32) #Matrix of trigrams consisting of one-hot encoded words.
onehot_lefts = onehot_trigrams[:,0] #The left side of the trigram.
onehot_rights = onehot_trigrams[:,2] #The right side of the trigram.
onehot_targets = onehot_trigrams[:,1] #The middle of the trigram.

###################################

class Model(object):

    def __init__(self, vocab_size):
        learning_rate = 0.1
        momentum = 0.9
        init_stddev = 1e-2
        hidden_layer_size = 4
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.lefts   = tf.placeholder(tf.float32, [None, vocab_size], 'lefts') #Left word in trigram as a one-hot vector.
            self.rights  = tf.placeholder(tf.float32, [None, vocab_size], 'rights') #Right word in trigram as a one-hot vector.
            self.targets = tf.placeholder(tf.int32, [None], 'targets') #Middle word in trigram as an index.

            self.params = []
            
            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', [vocab_size, hidden_layer_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [hidden_layer_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                
                #The context of the middle word is the left and right words together, concatenated into a single vector.
                context = tf.concat([ self.lefts, self.rights ], axis=1)
                hs = tf.sigmoid(tf.matmul(context, W) + b)

            with tf.variable_scope('output'):
                #Predict the middle word from the context.
                W = tf.get_variable('W', [hidden_layer_size, vocab_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [vocab_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                logits = tf.matmul(hs, W) + b
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
    
    def predict_hidden(self, onehots):
        return self.sess.run([ self.hs_lefts ], { self.lefts: onehots })[0]

###################################

max_epochs = 1000

(fig, ax) = plt.subplots(1, 1)

[ train_error_plot ] = axs[1].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
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

model = Model(vocab_size)
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(onehot_lefts, onehot_rights, index_targets)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(onehot_lefts, onehot_rights, index_targets)

print()
probs = model.predict(onehot_lefts, onehot_rights)
trigrams_shown = set()
print('3gram', 'top 3 predicted middle tokens', sep='\t')
for (trigram, ps) in zip(token_trigrams, probs):
    if trigram not in trigrams_shown: 
        top_probs = sorted(zip(ps, vocab), reverse=True)[:3]
        print(' '.join(trigram), ' '.join([ '{} ({:.5f})'.format(t, p) for (p, t) in top_probs ]), sep='\t')
        trigrams_shown.add(trigram)

model.close()