import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None, 2], 'xs')
    ts = tf.placeholder(tf.float32, [None, 1], 'ts')
    dropout = tf.placeholder(tf.bool, [], 'dropout') #Add a placeholder for whether to use dropout (during training) or not (during predictions)

    #Decide on the dropout keep-probability depending on whether to use dropout or not where a keep-probability of 1 means that no dropout is used
    dropout_keep_prob = tf.cond(dropout, lambda:tf.constant(1.0-0.125, tf.float32), lambda:tf.constant(1.0, tf.float32))
    
    hidden_layer_size = 8
    
    with tf.variable_scope('hidden'):
        W = tf.get_variable('W', [2, hidden_layer_size], tf.float32, tf.random_normal_initializer(stddev=0.2, seed=0))
        b = tf.get_variable('b', [hidden_layer_size], tf.float32, tf.zeros_initializer())
        hs = tf.nn.dropout(tf.sigmoid(tf.matmul(xs, W) + b), dropout_keep_prob) #Apply dropout on the hidden layer using the above keep-probability

    with tf.variable_scope('output'):
        W = tf.get_variable('W', [hidden_layer_size, 1], tf.float32, tf.random_normal_initializer(stddev=0.2, seed=0))
        b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
        ys = tf.sigmoid(tf.matmul(hs, W) + b)
    
    error = tf.reduce_mean((ys - ts)**2)
    
    step = tf.train.MomentumOptimizer(1.0, 0.5).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_x = [[0,0], [0,1], [1,0], [1,1]]
        train_y = [ [0],   [1],   [1],   [0] ]
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 1000+1):
            s.run([ step ], { xs: train_x, ts: train_y, dropout: True })

            [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y, dropout: False })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error)
                
                (all_x0s, all_x1s) = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
                [ all_ys ] = s.run([ ys ], { xs: np.stack([np.reshape(all_x0s, [-1]), np.reshape(all_x1s, [-1])], axis=1), dropout: False })
                all_ys = np.reshape(all_ys, [50, 50])
                
                ax[0].cla()
                ax[0].contourf(all_x0s, all_x1s, all_ys, 100, vmin=0.0, vmax=1.0, cmap='bwr')
                ax[0].set_xlim(0.0, 1.0)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(0.0, 1.0)
                ax[0].set_ylabel('x1')
                ax[0].set_title('Neural network')
                ax[0].grid(True)
                
                ax[1].cla()
                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].set_xlim(0, 1000)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0.0, 0.26)
                ax[1].set_ylabel('MSE')
                ax[1].grid(True)
                ax[1].set_title('Error progress')
                ax[1].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        fig.show()
