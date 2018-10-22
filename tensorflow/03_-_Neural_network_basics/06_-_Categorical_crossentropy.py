import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None, 2], 'xs')
    ts = tf.placeholder(tf.int32, [None], 'ts')

    W = tf.get_variable('W', [2, 4], tf.float32, tf.zeros_initializer())
    b = tf.get_variable('b', [4], tf.float32, tf.zeros_initializer())
    logits = tf.matmul(xs, W) + b #The logits are going to be used for the softmax's cross entropy
    ys = tf.nn.softmax(logits)
    
    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ts, logits=logits))
    
    step = tf.train.GradientDescentOptimizer(1.0).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 5, gridspec_kw={'width_ratios': [1,1,1,1,4]})
        plt.ion()
        
        train_x = [[0,0], [0,1], [1,0], [1,1]]
        train_y = [  0,     1,     2,     3 ]
        
        train_errors = list()
        print('epoch', 'trainerror')
        for epoch in range(1, 500+1):
            indexes = np.arange(len(train_x))
            np.random.shuffle(indexes)
            for i in range(len(indexes)):
                s.run([ step ], { xs: [train_x[i]], ts: [train_y[i]] })

            [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error)
                
                ax[0].cla()
                ax[1].cla()
                ax[2].cla()
                ax[3].cla()
                ax[4].cla()
                
                (all_x0s, all_x1s) = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
                [ all_ys ] = s.run([ ys ], { xs: np.stack([np.reshape(all_x0s, [-1]), np.reshape(all_x1s, [-1])], axis=1) })
                all_ys = np.reshape(all_ys, [50, 50, 4])
                
                ax[0].contourf(all_x0s, all_x1s, all_ys[:,:,0], 100, vmin=0.0, vmax=1.0, cmap='bwr')
                ax[0].set_xlim(0.0, 1.0)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(0.0, 1.0)
                ax[0].set_ylabel('x1')
                ax[0].grid(True)
                ax[0].set_title('Multiplexer 1')

                ax[1].contourf(all_x0s, all_x1s, all_ys[:,:,1], 100, vmin=0.0, vmax=1.0, cmap='bwr')
                ax[1].set_xlim(0.0, 1.0)
                ax[1].set_xlabel('x0')
                ax[1].set_ylim(0.0, 1.0)
                ax[1].grid(True)
                ax[1].set_title('Multiplexer 2')

                ax[2].contourf(all_x0s, all_x1s, all_ys[:,:,2], 100, vmin=0.0, vmax=1.0, cmap='bwr')
                ax[2].set_xlim(0.0, 1.0)
                ax[2].set_xlabel('x0')
                ax[2].set_ylim(0.0, 1.0)
                ax[2].grid(True)
                ax[2].set_title('Multiplexer 3')

                ax[3].contourf(all_x0s, all_x1s, all_ys[:,:,3], 100, vmin=0.0, vmax=1.0, cmap='bwr')
                ax[3].set_xlim(0.0, 1.0)
                ax[3].set_xlabel('x0')
                ax[3].set_ylim(0.0, 1.0)
                ax[3].grid(True)
                ax[3].set_title('Multiplexer 4')
                
                ax[4].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[4].set_xlim(-10, 500)
                ax[4].set_xlabel('epoch')
                ax[4].set_ylim(0.0, 1.0)
                ax[4].set_ylabel('XE')
                ax[4].grid(True)
                ax[4].set_title('Error progress')
                ax[4].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        fig.show()
