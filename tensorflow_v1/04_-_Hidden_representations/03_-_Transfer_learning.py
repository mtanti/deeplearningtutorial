import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

#Training set of sentences that are labeled with their part-of-speech tags.
token_sents = [
        '<BEG> the dog barks and the cat meows <END>'.split(' '),
        '<BEG> the cat meows and the dog barks <END>'.split(' '),
    ]
tag_sents = [
        'EDGE DET NOUN VERB CONJ DET NOUN VERB EDGE'.split(' '),
        'EDGE DET NOUN VERB CONJ DET NOUN VERB EDGE'.split(' '),
    ]

#Dataset for word predictor.

token_trigrams = [ tuple(sent[i:i+3]) for sent in token_sents for i in range(len(sent)-2) ]

vocab = sorted({ token for sent in token_sents for token in sent })

token2index = { token: index for (index, token) in enumerate(vocab) }
index_sents = [ [ token2index[token] for token in sent ] for sent in token_sents ]
index_trigrams = np.array([ sent[i:i+3] for sent in index_sents for i in range(len(sent)-2) ], np.int32)
index_lefts = index_trigrams[:,0]
index_rights = index_trigrams[:,2]
index_targets = index_trigrams[:,1]

#Dataset for tagger.

tag_trigrams = [ tuple(sent[i:i+3]) for sent in tag_sents for i in range(len(sent)-2) ]

tags = sorted({ tag for sent in tag_sents for tag in sent })

tag2index = { tag: index for (index, tag) in enumerate(tags) }
index_tags = [ [ tag2index[tag] for tag in sent ] for sent in tag_sents ]
index_target_tags = np.array([ sent[i] for sent in index_tags for i in range(1, len(sent)-1) ], np.int32) #Tag of the middle word in each trigram.

###################################

class WordPredictorModel(object):

    def __init__(self, vocab_size):
        learning_rate = 0.1
        momentum = 0.9
        init_stddev = 1e-2
        embed_size = 2
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.lefts   = tf.placeholder(tf.int32, [None], 'lefts')
            self.rights  = tf.placeholder(tf.int32, [None], 'rights')
            self.targets = tf.placeholder(tf.int32, [None], 'targets')

            self.params = []
            
            with tf.variable_scope('embeddings'):
                self.embedding_matrix = tf.get_variable('embedding_matrix', [ vocab_size, embed_size ], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.params.extend([ self.embedding_matrix ])

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

class TaggerModel(object):

    def __init__(self, embedding_matrix, finetune, num_tags):
        learning_rate = 0.1
        momentum = 0.9
        init_stddev = 1e-2
        
        vocab_size = embedding_matrix.shape[0]
        embed_size = embedding_matrix.shape[1]
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.phrases = tf.placeholder(tf.int32, [None, 3], 'phrases')
            self.targets = tf.placeholder(tf.int32, [None], 'targets')

            batch_size = tf.shape(self.phrases)[0] #The number of phrases passed in as a batch.
            self.params = []
            
            with tf.variable_scope('embeddings'):
                self.embedding_matrix = tf.get_variable('embedding_matrix', [ vocab_size, embed_size ], tf.float32, tf.constant_initializer(embedding_matrix), trainable=finetune) #Use the provided embedding matrix as an initial value for the embedding matrix variable and set whether you want to train it or not.
                self.params.extend([ self.embedding_matrix ])

                embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.phrases)
                embedded = tf.reshape(embedded, [batch_size, 3*embed_size]) #Join the embedding vectors of the three words into one.

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [3*embed_size, num_tags], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [num_tags], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                logits = tf.matmul(embedded, W) + b
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
    
    def optimisation_step(self, phrases, targets):
        return self.sess.run([ self.optimiser_step ], { self.phrases: phrases, self.targets: targets })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_error(self, phrases, targets):
        return self.sess.run([ self.error ], { self.phrases: phrases, self.targets: targets })[0]
    
    def predict(self, phrases):
        return self.sess.run([ self.probs ], { self.phrases: phrases })[0]
    
    def get_embeddings(self):
        return self.sess.run([ self.embedding_matrix ], { })[0]

###################################

max_epochs_source = 500
max_epochs_target = 50

(fig, axs) = plt.subplots(1, 3)

[ random_train_error_plot ] = axs[0].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[0].set_xlim(0, max_epochs_target)
axs[0].set_xlabel('epoch')
axs[0].set_ylim(0.0, 2.0)
axs[0].set_ylabel('XE')
axs[0].grid(True)
axs[0].set_title('Random')
axs[0].legend()

[ frozen_train_error_plot ] = axs[1].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[1].set_xlim(0, max_epochs_target)
axs[1].set_xlabel('epoch')
axs[1].set_ylim(0.0, 2.0)
axs[1].set_ylabel('XE')
axs[1].grid(True)
axs[1].set_title('Frozen')
axs[1].legend()

[ finetuned_train_error_plot ] = axs[2].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
axs[2].set_xlim(0, max_epochs_target)
axs[2].set_xlabel('epoch')
axs[2].set_ylim(0.0, 2.0)
axs[2].set_ylabel('XE')
axs[2].grid(True)
axs[2].set_title('Fine-tuned')
axs[2].legend()

fig.tight_layout()
fig.show()

###################################

print('Training source model')

model = WordPredictorModel(len(vocab))
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs_source+1):
    train_error = model.get_error(index_lefts, index_rights, index_targets)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
    
    model.optimisation_step(index_lefts, index_rights, index_targets)

embedding_matrix = model.get_embeddings()

model.close()

print()

###################################

print('Random')

model = TaggerModel(np.random.normal(0, 1e-2, size=embedding_matrix.shape), True, len(tags))
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs_target+1):
    train_error = model.get_error(index_trigrams, index_target_tags)
    train_errors.append(train_error)
    
    if epoch%10 == 0:
        print(epoch, train_error, sep='\t')
        
        random_train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(index_trigrams, index_target_tags)

model.close()

print()

###################################

print('Frozen')

model = TaggerModel(embedding_matrix, False, len(tags))
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs_target+1):
    train_error = model.get_error(index_trigrams, index_target_tags)
    train_errors.append(train_error)
    
    if epoch%10 == 0:
        print(epoch, train_error, sep='\t')
        
        frozen_train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(index_trigrams, index_target_tags)

model.close()

print()

###################################

print('Fine-tuned')

model = TaggerModel(embedding_matrix, True, len(tags))
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs_target+1):
    train_error = model.get_error(index_trigrams, index_target_tags)
    train_errors.append(train_error)
    
    if epoch%10 == 0:
        print(epoch, train_error, sep='\t')
        
        finetuned_train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(index_trigrams, index_target_tags)

print()
probs = model.predict(index_trigrams)
trigrams_shown = set()
print('3gram', 'middle tag', 'top 3 predicted middle tag', sep='\t')
for (trigram, tag_trigram, ps) in zip(token_trigrams, tag_trigrams, probs):
    if trigram not in trigrams_shown: 
        top_probs = sorted(zip(ps, tags), reverse=True)[:3]
        print(' '.join(trigram), tag_trigram[1], ' '.join([ '{} ({:.5f})'.format(t, p) for (p, t) in top_probs ]), sep='\t')
        trigrams_shown.add(trigram)
        
model.close()