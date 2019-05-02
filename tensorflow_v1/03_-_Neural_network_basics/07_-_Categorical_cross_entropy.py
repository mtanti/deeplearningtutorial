import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 1.0
max_epochs = 500

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None, 2], 'xs')
    ts = tf.placeholder(tf.int32, [None], 'ts')

    W = tf.get_variable('W', [2, 4], tf.float32, tf.zeros_initializer())
    b = tf.get_variable('b', [4], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(xs, W) + b
    ys = tf.nn.softmax(logits)
    
    #Taking the mean of the categorical cross entropy error of each training set item
    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ts, logits=logits))
    
    step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig1, ax1) = plt.subplots(2, 2)
        (fig2, ax2) = plt.subplots(1, 1)
        plt.ion()
        
        #Training set for a binary decoder (convert binary number into index)
        train_x = [ [0,0], [0,1], [1,0], [1,1] ]
        train_y = [   0,     1,     2,     3   ]
        
        train_errors = list()
        print('epoch', 'train error', sep='\t')
        for epoch in range(1, max_epochs+1):
            s.run([ step ], { xs: train_x, ts: train_y })

            [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error, sep='\t')
                
                (all_x0s, all_x1s) = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
                [ all_ys ] = s.run([ ys ], { xs: np.stack([np.reshape(all_x0s, [-1]), np.reshape(all_x1s, [-1])], axis=1) })
                all_ys = np.reshape(all_ys, [50, 50, 4])
                
                decoder = 0
                for row in range(2):
                    for col in range(2):
                        ax1[row, col].cla()
                        ax1[row, col].contourf(all_x0s, all_x1s, all_ys[:,:,decoder], 100, vmin=0.0, vmax=1.0, cmap='bwr')
                        ax1[row, col].set_xlim(0.0, 1.0)
                        ax1[row, col].set_xlabel('x0')
                        ax1[row, col].set_ylim(0.0, 1.0)
                        ax1[row, col].set_ylabel('x1')
                        ax1[row, col].grid(True)
                        ax1[row, col].set_title('Decoder '+str(decoder))
                        decoder += 1
                
                ax2.cla()
                ax2.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax2.set_xlim(0, max_epochs)
                ax2.set_xlabel('epoch')
                ax2.set_ylim(0.0, 1.0)
                ax2.set_ylabel('XE') #Cross entropy
                ax2.grid(True)
                ax2.set_title('Error progress')
                ax2.legend()
                
                fig1.tight_layout()
                fig2.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        fig1.show()
        fig2.show()