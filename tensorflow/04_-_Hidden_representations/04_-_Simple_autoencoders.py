import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    #Only inputs are needed as the output will be the input again
    xs = tf.placeholder(tf.float32, [None, 4], 'xs')

    hidden_layer_size = 2
    
    with tf.variable_scope('encoder'):
        W = tf.get_variable('W', [4, hidden_layer_size], tf.float32, tf.random_normal_initializer(stddev=0.1, seed=0))
        b = tf.get_variable('b', [hidden_layer_size], tf.float32, tf.zeros_initializer())
        hs = tf.sigmoid(tf.matmul(xs, W) + b)
        #Hidden layer is a vector that is smaller than the input in order to force compression

    with tf.variable_scope('decoder'):
        W = tf.get_variable('W', [hidden_layer_size, 4], tf.float32, tf.random_normal_initializer(stddev=0.1, seed=0))
        b = tf.get_variable('b', [4], tf.float32, tf.zeros_initializer())
        logits = tf.matmul(hs, W) + b
        ys = tf.sigmoid(logits)
    
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=xs))
    
    step = tf.train.MomentumOptimizer(1.0, 0.5).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_x = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 1000+1):
            s.run([ step ], { xs: train_x })

            [ train_error ] = s.run([ error ], { xs: train_x })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error)
                
                [ curr_hs ] = s.run([ hs ], { xs: train_x })

                ax[0].cla()
                for (curr_h, curr_x) in zip(curr_hs.tolist(), train_x):
                    ax[0].plot(curr_h[0], curr_h[1], linestyle='', marker='o', markersize=10)
                    ax[0].text(curr_h[0], curr_h[1], '{}{}{}{}'.format(*curr_x))
                ax[0].set_xlim(-0.1, 1.1)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(-0.1, 1.1)
                ax[0].set_ylabel('x1')
                ax[0].set_title('Hidden layer')
                ax[0].grid(True)

                ax[1].cla()
                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].set_xlim(0, 1000)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0.0, 0.5)
                ax[1].set_ylabel('XE') #Cross entropy
                ax[1].grid(True)
                ax[1].set_title('Error progress')
                ax[1].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        fig.show()

        print()
        print('xs to ys')
        [ curr_ys ] = s.run([ ys ], { xs: train_x })
        for (curr_x, curr_y) in zip(train_x, curr_ys.tolist()):
            print(curr_x, np.round(curr_y, 3))
            
        print()
        print('xs to hs (encoder)')
        [ curr_hs ] = s.run([ hs ], { xs: train_x })
        for (curr_x, curr_h) in zip(train_x, curr_hs.tolist()):
            print(curr_x, np.round(curr_h, 3))

        print()
        print('hs to ys (decoder)')
        curr_hs = [[0,0], [0,1], [1,0], [1,1]]
        [ curr_ys ] = s.run([ ys ], { hs: curr_hs })
        for (curr_h, curr_y) in zip(curr_hs, curr_ys.tolist()):
            print(curr_h, np.round(curr_y, 3))

        print()
        print('average hs')
        [ curr_hs ] = s.run([ hs ], { xs: [[1,0,0,0], [0,0,0,1]] })
        [ curr_ys ] = s.run([ ys ], { hs: [np.mean(curr_hs, axis=0)] })
        print('dec(mean([enc([1,0,0,0]), enc([0,0,0,1])])) =', np.round(curr_ys, 3))
