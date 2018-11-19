import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None, 4], 'xs')

    #Encoder creates a random normal encoding using a mean and a standard deviation
    with tf.variable_scope('encoder'):
        W_mean = tf.get_variable('W_mean', [4, 2], tf.float32, tf.zeros_initializer())
        b_mean = tf.get_variable('b_mean', [2], tf.float32, tf.zeros_initializer())
        hs_mean = tf.matmul(xs, W_mean) + b_mean

        W_stddev = tf.get_variable('W_stddev', [4, 2], tf.float32, tf.zeros_initializer())
        b_stddev = tf.get_variable('b_stddev', [2], tf.float32, tf.ones_initializer())
        hs_log_stddev = tf.matmul(xs, W_stddev) + b_stddev #Standard deviation needs to be positive so assume that this is the log of the stddev

        #Generate the random encoding
        hs = tf.exp(hs_log_stddev)*tf.random_normal(tf.shape(hs_mean), seed=0) + hs_mean

    with tf.variable_scope('decoder'):
        W = tf.get_variable('W', [2, 4], tf.float32, tf.random_normal_initializer(stddev=0.1, seed=0))
        b = tf.get_variable('b', [4], tf.float32, tf.zeros_initializer())
        logits = tf.matmul(hs, W) + b
        ys = tf.sigmoid(logits)
    
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=xs))
    
    #This is a measure of how far the encoding is from being a standard normal distribution (mean=0, stddev=1)
    kl_divergence = tf.reduce_mean(1.0 + hs_log_stddev - tf.square(hs_mean) - tf.exp(hs_log_stddev))
    
    loss = error - 0.2*kl_divergence
    
    step = tf.train.MomentumOptimizer(0.1, 0.5).minimize(loss)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_x = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 2000+1):
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
                ax[0].set_xlim(-5.0, 5.0)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(-5.0, 5.0)
                ax[0].set_ylabel('x1')
                ax[0].set_title('Hidden layer')
                ax[0].grid(True)

                ax[1].cla()
                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].set_xlim(0, 2000)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0.0, 2.0)
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
        print('xs to means')
        [ curr_hs_mean ] = s.run([ hs_mean ], { xs: train_x })
        for (curr_x, curr_h_mean) in zip(train_x, curr_hs_mean.tolist()):
            print(curr_x, np.round(curr_h_mean, 3))

        print()
        print('xs to stddevs')
        [ curr_hs_log_stddev ] = s.run([ hs_log_stddev ], { xs: train_x })
        for (curr_x, curr_h_log_stddev) in zip(train_x, curr_hs_log_stddev.tolist()):
            print(curr_x, np.round(np.exp(curr_h_log_stddev), 3))

        print()
        print('random hs to ys (decoder)')
        for _ in range(5):
            curr_h = np.random.normal(size=[2])
            [ curr_y ] = s.run([ ys ], { hs: [curr_h] })
            print(curr_h, np.round(curr_y, 3))

        print()
        print('average hs')
        [ curr_hs ] = s.run([ hs ], { xs: [[1,0,0,0], [0,0,0,1]] })
        [ curr_ys ] = s.run([ ys ], { hs: [np.mean(curr_hs, axis=0)] })
        print('dec(mean([enc([1,0,0,0]), enc([0,0,0,1])])) =', np.round(curr_ys, 3))
