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
                [ 1.0 ] if px == '#' else [ 0 ]
                for px in row
            ] for row in img
        ] for img in char_images
    ], np.float32)

###################################

class Model(object):

    def __init__(self):
        learning_rate = 1.0
        momentum = 0.9
        init_stddev = 1e-2
        embed_size = 2
        kernel1_width = 2
        kernel1_height = 2
        kernel1_size = 2
        kernel2_width = 2
        kernel2_height = 2
        kernel2_size = 2
        image_width = 3
        image_height = 3
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images  = tf.placeholder(tf.float32, [None, image_height, image_width, 1], 'images')
            self.targets = tf.placeholder(tf.float32, [None, 1], 'targets')

            self.params = []

            batch_size = tf.shape(self.images)[0]
            
            with tf.variable_scope('hidden1'):
                W = tf.get_variable('W', [kernel1_height, kernel1_width, 1, kernel1_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [kernel1_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.conv_hs1 = tf.sigmoid(tf.nn.conv2d(self.images, W, [1,1,1,1], 'VALID') + b)

                num_slides_x_per_img = image_width - kernel1_width + 1
                num_slides_y_per_img = image_height - kernel1_height + 1

            with tf.variable_scope('hidden2'):
                W = tf.get_variable('W', [kernel2_height, kernel2_width, kernel1_size, kernel2_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [kernel2_size], tf.float32, tf.zeros_initializer())
                self.params.extend([ W, b ])
                self.conv_hs2 = tf.sigmoid(tf.nn.conv2d(self.conv_hs1, W, [1,1,1,1], 'VALID') + b)

                num_slides_x_per_img = num_slides_x_per_img - kernel2_width + 1
                num_slides_y_per_img = num_slides_y_per_img - kernel2_height + 1
                vec_size_per_img = (num_slides_x_per_img*num_slides_y_per_img)*kernel2_size
                self.flat_hs2 = tf.reshape(self.conv_hs2, [batch_size, vec_size_per_img])

            with tf.variable_scope('output'):
                W = tf.get_variable('W', [vec_size_per_img, 1], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
                logits = tf.matmul(self.flat_hs2, W) + b
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
    
###################################

max_epochs = 2000

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

model = Model()
model.initialise()

train_errors = list()
print('epoch', 'train error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(bin_images, lines)
    train_errors.append(train_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, sep='\t')
        
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        plt.draw()
        fig.canvas.flush_events()
    
    model.optimisation_step(bin_images, lines)

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