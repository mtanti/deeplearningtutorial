import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 1.0
max_epochs = 500

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None, 2], 'xs')
    ts = tf.placeholder(tf.float32, [None, 1], 'ts')

    W = tf.get_variable('W', [2, 1], tf.float32, tf.zeros_initializer())
    b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(xs, W) + b
    ys = tf.sigmoid(logits)
    
    #Taking the mean of the binary cross entropy error of each training set item
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ts, logits=logits))
    
    step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 1)
        plt.ion()
        
        #Training set for an OR function
        train_x = [ [0,0], [0,1], [1,0], [1,1] ]
        train_y = [  [0],   [1],   [1],   [1]  ]
        
        train_errors = list()
        print('epoch', 'train error', sep='\t')
        for epoch in range(1, max_epochs+1):
            s.run([ step ], { xs: train_x, ts: train_y })

            [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error, sep='\t')
                
                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax.set_xlim(0, max_epochs)
                ax.set_xlabel('epoch')
                ax.set_ylim(0.0, 0.26)
                ax.set_ylabel('XE') #Cross entropy
                ax.grid(True)
                ax.set_title('Error progress')
                ax.legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        fig.show()
