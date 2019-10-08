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
tag_sents = [
        'EDGE DET NOUN VERB CONJ DET NOUN VERB EDGE'.split(' '),
        'EDGE DET NOUN VERB CONJ DET NOUN VERB EDGE'.split(' '),
    ]

token_trigrams = [ tuple(sent[i:i+3]) for sent in token_sents for i in range(len(sent)-2) ]

vocab = sorted({ token for sent in token_sents for token in sent })

token2index = { token: index for (index, token) in enumerate(vocab) }
index_sents = [ [ token2index[token] for token in sent ] for sent in token_sents ]
index_trigrams = np.array([ sent[i:i+3] for sent in index_sents for i in range(len(sent)-2) ], np.int32)
index_lefts = index_trigrams[:,0]
index_rights = index_trigrams[:,2]
index_targets = index_trigrams[:,1]

tag_trigrams = [ tuple(sent[i:i+3]) for sent in tag_sents for i in range(len(sent)-2) ]

tags = sorted({ tag for sent in tag_sents for tag in sent })

tag2index = { tag: index for (index, tag) in enumerate(tags) }
index_tags = [ [ tag2index[tag] for tag in sent ] for sent in tag_sents ]
index_target_tags = np.array([ sent[i] for sent in index_tags for i in range(1, len(sent)-1) ], np.int32)

###################################

class Model(object):

    def __init__(self, vocab_size, num_tags):
        learning_rate = 0.1
        momentum = 0.9
        init_stddev = 1e-2
        embed_size = 2
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.params = []
            
            #Shared variables.
            with tf.variable_scope('embeddings'):
                self.embedding_matrix = tf.get_variable('embedding_matrix', [ vocab_size, embed_size ], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.params.extend([ self.embedding_matrix ])
            
            #Word predictor task.
            with tf.variable_scope('word_predictor'):
                self.lefts = tf.placeholder(tf.int32, [None], 'lefts')
                self.rights = tf.placeholder(tf.int32, [None], 'rights')
                self.word_targets = tf.placeholder(tf.int32, [None], 'word_targets')

                with tf.variable_scope('embeddings'):
                    embedded_lefts  = tf.nn.embedding_lookup(self.embedding_matrix, self.lefts)
                    embedded_rights = tf.nn.embedding_lookup(self.embedding_matrix, self.rights)
                    embedded_context = tf.concat([ embedded_lefts, embedded_rights ], axis=1)

                with tf.variable_scope('output'):
                    W = tf.get_variable('W', [2*embed_size, vocab_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                    b = tf.get_variable('b', [vocab_size], tf.float32, tf.zeros_initializer())
                    self.params.extend([ W, b ])
                    logits = tf.matmul(embedded_context, W) + b
                    self.word_probs = tf.nn.softmax(logits)
                
                self.word_error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.word_targets, logits=logits))
            
            #Tagger task.
            with tf.variable_scope('tagger'):
                self.phrases = tf.placeholder(tf.int32, [None, 3], 'phrases')
                self.tag_targets = tf.placeholder(tf.int32, [None], 'tag_targets')
                
                batch_size = tf.shape(self.phrases)[0]
                
                with tf.variable_scope('embeddings'):
                    embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.phrases)
                    embedded = tf.reshape(embedded, [batch_size, 3*embed_size])

                with tf.variable_scope('output'):
                    W = tf.get_variable('W', [3*embed_size, num_tags], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                    b = tf.get_variable('b', [num_tags], tf.float32, tf.zeros_initializer())
                    self.params.extend([ W, b ])
                    logits = tf.matmul(embedded, W) + b
                    
                    self.tag_probs = tf.nn.softmax(logits)
                    
                self.tag_error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tag_targets, logits=logits))
            
            #Multi-objective optimisation.
            self.error = self.word_error + self.tag_error
            
            self.optimiser_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.error)
        
            self.init = tf.global_variables_initializer()
            
            self.graph.finalize()

            self.sess = tf.Session()
    
    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, lefts, rights, word_targets, phrases, tag_targets):
        return self.sess.run([ self.optimiser_step ], { self.lefts: lefts, self.rights: rights, self.word_targets: word_targets, self.phrases: phrases, self.tag_targets: tag_targets })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_errors(self, lefts, rights, word_targets, phrases, tag_targets):
        return self.sess.run([ self.error, self.word_error, self.tag_error ], { self.lefts: lefts, self.rights: rights, self.word_targets: word_targets, self.phrases: phrases, self.tag_targets: tag_targets })
    
    def predict_words(self, lefts, rights):
        return self.sess.run([ self.word_probs ], { self.lefts: lefts, self.rights: rights })[0]
    
    def predict_tags(self, phrases):
        return self.sess.run([ self.tag_probs ], { self.phrases: phrases })[0]

###################################

max_epochs = 100

(fig, ax) = plt.subplots(1, 1)

[ train_error_plot ] = ax.plot([], [], color='purple', linestyle='-', linewidth=2, label='train')
[ train_word_error_plot ] = ax.plot([], [], color='blue', linestyle='-', linewidth=1, label='train_word')
[ train_tag_error_plot ] = ax.plot([], [], color='red', linestyle='-', linewidth=1, label='train_tag')
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

model = Model(len(vocab), len(tags))
model.initialise()

train_errors = list()
train_word_errors = list()
train_tag_errors = list()
print('epoch', 'train error', 'train word error', 'train tag error', sep='\t')
for epoch in range(1, max_epochs+1):
    [ train_error, train_word_error, train_tag_error ] = model.get_errors(index_lefts, index_rights, index_targets, index_trigrams, index_target_tags)
    train_errors.append(train_error)
    train_word_errors.append(train_word_error)
    train_tag_errors.append(train_tag_error)
    
    if epoch%10 == 0:
        print(epoch, train_error, train_word_error, train_tag_error, sep='\t')
        
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        train_word_error_plot.set_data(np.arange(len(train_errors)), train_word_errors)
        train_tag_error_plot.set_data(np.arange(len(train_errors)), train_tag_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(index_lefts, index_rights, index_targets, index_trigrams, index_target_tags)

print()
print('word predictor')
probs = model.predict_words(index_lefts, index_rights)
trigrams_shown = set()
print('3gram', 'top 3 predicted middle tokens', sep='\t')
for (trigram, ps) in zip(token_trigrams, probs):
    if trigram not in trigrams_shown: 
        top_probs = sorted(zip(ps, vocab), reverse=True)[:3]
        print(' '.join(trigram), ' '.join([ '{} ({:.5f})'.format(t, p) for (p, t) in top_probs ]), sep='\t')
        trigrams_shown.add(trigram)

print()
print('tagger')
probs = model.predict_tags(index_trigrams)
trigrams_shown = set()
print('3gram', 'middle tag', 'top 3 predicted middle tag', sep='\t')
for (trigram, tag_trigram, ps) in zip(token_trigrams, tag_trigrams, probs):
    if trigram not in trigrams_shown: 
        top_probs = sorted(zip(ps, tags), reverse=True)[:3]
        print(' '.join(trigram), tag_trigram[1], ' '.join([ '{} ({:.5f})'.format(t, p) for (p, t) in top_probs ]), sep='\t')
        trigrams_shown.add(trigram)

model.close()