import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None, 2], 'xs')
    ts = tf.placeholder(tf.float32, [None, 1], 'ts')

    W = tf.get_variable('W', [2, 1], tf.float32, tf.zeros_initializer())
    b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(xs, W) + b #The logits are going to be used for the softmax's cross entropy
    ys = tf.sigmoid(logits)
    
    error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ts, logits=logits))
    
    step = tf.train.GradientDescentOptimizer(1.0).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 1)
        plt.ion()
        
        train_x = [[0,0], [0,1], [1,0], [1,1]]
        train_y = [ [0],   [1],   [1],   [1] ]
        
        train_errors = list()
        print('epoch', 'train error')
        for epoch in range(1, 500+1):
            s.run([ step ], { xs: train_x, ts: train_y })

            [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error)
                
                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax.set_xlim(-10, 500)
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
