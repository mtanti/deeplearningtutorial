import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

char_images = [
           ['###', #line
            '   ',
            '   '],

           ['   ', #line
            '###',
            '   '],

           ['   ', #line
            '   ',
            '###'],

           ['#  ', #line
            '#  ',
            '#  '],

           [' # ', #line
            ' # ',
            ' # '],

           ['  #', #line
            '  #',
            '  #'],

           ['   ', #no line
            '   ',
            '   '],

           ['#  ', #no line
            '   ',
            '  #'],

           ['   ', #no line
            ' # ',
            '   '],

           ['   ', #no line
            '   ',
            '  #'],

           ['  #', #no line
            ' ##',
            '#  '],

           ['# #', #no line
            ' # ',
            ' # '],

           [' # ', #no line
            '#  ',
            '# #'],

           ['## ', #no line
            '   ',
            '  #'],

           ['#  ', #no line
            ' ##',
            '   '],

           ['###', #no line
            '###',
            '###'],
    ]
lines = [
        [ 1 ],
        [ 1 ],
        [ 1 ],
        [ 1 ],
        [ 1 ],
        [ 1 ],
        [ 0 ],
        [ 0 ],
        [ 0 ],
        [ 0 ],
        [ 0 ],
        [ 0 ],
        [ 0 ],
        [ 0 ],
        [ 0 ],
        [ 0 ],
    ]

bin_images = np.array([
        [
            [
                [ 1.0 ] if px == '#' else [ 0 ] #Make the image pixels consist of single element vectors.
                for px in row
            ] for row in img
        ] for img in char_images
    ], np.float32)

#Like the indexed bigrams, but instead this is a bunch of 2x2 image regions.
bin_regions = np.unique(np.concatenate([ bin_images[:, i:i+2, j:j+2,:] for i in range(2) for j in range(2) ]), axis=0)

#Represent regions as binary numbers to make it easier to show them on charts.
char_regions = [
        ''.join('1' if b == [1.0] else '0' for row in region for b in row)
        for region in bin_regions.tolist()
    ]

###################################

class Model(object):

    def __init__(self):
        learning_rate = 1.0
        momentum = 0.9
        init_stddev = 1e-2
        embed_size = 2
        kernel_width = 2
        kernel_height = 2
        kernel_size = 2
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images  = tf.placeholder(tf.float32, [None, None, None, 1], 'images')
            self.targets = tf.placeholder(tf.float32, [None, 1], 'targets')

            self.params = []

            batch_size = tf.shape(self.images)[0]
            
            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', [kernel_height, kernel_width, 1, kernel_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev)) #Note that the image consists of a grid of single element (1) vectors.
                b = tf.get_variable('b', [kernel_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.conv_hs = tf.sigmoid(tf.nn.conv2d(self.images, W, [1,1,1,1], 'VALID') + b)

                #Perform max pooling but first turn the resultant grid of vectors into a sequence in order to become a single vector after pooling.
                num_conv_y = tf.shape(self.conv_hs)[1]
                num_conv_x = tf.shape(self.conv_hs)[2]
                flat_hs = tf.reshape(self.conv_hs, [ batch_size, num_conv_y*num_conv_x, kernel_size ])
                self.pool_hs = tf.reduce_max(flat_hs, axis=1) #Max pooling

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
    
    def optimisation_step(self, images, targets):
        return self.sess.run([ self.optimiser_step ], { self.images: images, self.targets: targets })
    
    def get_params(self):
        return self.sess.run(self.params, { })
    
    def get_error(self, images, targets):
        return self.sess.run([ self.error ], { self.images: images, self.targets: targets })[0]
    
    def predict(self, images):
        return self.sess.run([ self.probs ], { self.images: images })[0]
    
    def get_conv(self, images):
        return self.sess.run([ self.conv_hs ], { self.images: images })[0]

    def get_pool(self, images):
        return self.sess.run([ self.pool_hs ], { self.images: images })[0]

###################################

max_epochs = 2000

(fig, axs) = plt.subplots(1, 2)

region_plots = list()
region_texts = list()
for char_region in char_regions:
    [ region_plot ] = axs[0].plot([ 0 ], [ 0 ], linestyle='', marker='o', markersize=10)
    region_plots.append(region_plot)
    region_text = axs[0].text(0, 0, char_region, fontdict={ 'fontsize': 8 })
    region_texts.append(region_text)
axs[0].set_xlim(0.0, 1.0)
axs[0].set_xlabel('d0')
axs[0].set_ylim(0.0, 1.0)
axs[0].set_ylabel('d1')
axs[0].grid(True)
axs[0].set_title('Regions')

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

model = Model()
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(bin_images, lines)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        
        convs = model.get_conv(bin_regions)
        
        for (region_plot, region_text, conv) in zip(region_plots, region_texts, convs.tolist()):
            region_plot.set_data([ conv[0][0][0] ], [ conv[0][0][1] ])
            region_text.set_position( (conv[0][0][0], conv[0][0][1]) )
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(bin_images, lines)

print()
print('region', 'vector', sep='\t')
convs = model.get_conv(bin_regions)
for (char_region, conv) in zip(char_regions, convs.tolist()):
    print(char_region, np.round(conv[0][0], 3), sep='\t')
print()

probs = model.predict(bin_images)
print('image/line')
for (char_image, prob) in zip(char_images, probs.tolist()):
    print('---')
    print('\n'.join(char_image))
    print('---')
    print(np.round(prob[0], 3), sep='\t')
    print()
    
model.close()