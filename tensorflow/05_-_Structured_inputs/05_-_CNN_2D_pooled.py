import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

img_width = 3
img_height = 3

pixels = [
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
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ]

binarised_pixels = np.array([
                [
                    [
                        [1] if px == '#' else [0] #Make the image pixels consist of single element vectors
                        for px in row
                    ] for row in img
                ] for img in pixels
            ])

binarised_windowed_pixels = np.unique(np.concatenate([ binarised_pixels[:,i:i+2,j:j+2,:] for i in range(2) for j in range(2) ]), axis=0)
            
windowed_pixels = [
                [
                    ''.join('1' if px == 1 else '0' for px in window)
                ] for window in binarised_windowed_pixels.reshape([-1, 2*2]).tolist()
            ]

g = tf.Graph()
with g.as_default():
    imgs = tf.placeholder(tf.float32, [None, None, None, 1], 'imgs')
    targets = tf.placeholder(tf.float32, [None, 1], 'targets')

    window_width = 2
    window_height = 2
    hidden_layer_size = 2
    W = tf.get_variable('W', [window_width, window_height, 1, hidden_layer_size], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    b = tf.get_variable('b', [hidden_layer_size], tf.float32, tf.zeros_initializer())
    conv_hs = tf.sigmoid(tf.nn.conv2d(imgs, W, [1,1,1,1], 'VALID') + b)

    num_windows_x_per_img = img_width - window_width + 1
    num_windows_y_per_img = img_height - window_height + 1
    
    hs = tf.contrib.layers.flatten(tf.reduce_max(tf.reshape(conv_hs, [-1, num_windows_x_per_img*num_windows_y_per_img, hidden_layer_size]), axis=1))
    
    W2 = tf.get_variable('W2', [hidden_layer_size, 1], tf.float32, tf.random_normal_initializer(stddev=0.01, seed=0))
    b2 = tf.get_variable('b2', [1], tf.float32)
    logits = tf.matmul(hs, W2) + b2
    probs = tf.sigmoid(logits)

    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    
    step = tf.train.GradientDescentOptimizer(1.0).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 5000+1):
            s.run([ step ], { imgs: binarised_pixels, targets: lines })

            [ train_error ] = s.run([ error ], { imgs: binarised_pixels, targets: lines })
            train_errors.append(train_error)
            
            if epoch%100 == 0:
                print(epoch, train_error)
                
                [ curr_window_vecs ] = s.run([ conv_hs ], { imgs: binarised_windowed_pixels })
                
                ax[0].cla()
                for (window, vector) in zip(windowed_pixels, curr_window_vecs[:,0,0,:].tolist()):
                    ax[0].plot(vector[0], vector[1], linestyle='', marker='o', markersize=10)
                    ax[0].text(vector[0], vector[1], ' '.join(window))
                ax[0].set_xlim(0, 1)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(0, 1)
                ax[0].set_ylabel('x1')
                ax[0].grid(True)
                ax[0].set_title('Windows')
                
                ax[1].cla()
                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].set_xlim(0, 5000)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0.0, 1.0)
                ax[1].set_ylabel('XE') #Cross entropy
                ax[1].grid(True)
                ax[1].set_title('Error progress')
                ax[1].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()
        
        [ curr_probs ] = s.run([ probs ], { imgs: binarised_pixels })
        for (img, prob) in zip(pixels, curr_probs[:,0].tolist()):
            print('\n'.join(img))
            print(round(prob, 2))
            print()
        fig.show()
