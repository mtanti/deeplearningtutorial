import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None, 2], 'xs')
    ts = tf.placeholder(tf.float32, [None, 1], 'ts')

    W = tf.get_variable('W', [2, 1], tf.float32, tf.zeros_initializer())
    b = tf.get_variable('b', [1], tf.float32, tf.zeros_initializer())
    ys = tf.sigmoid(tf.matmul(xs, W) + b)
    
    error = tf.reduce_mean((ys - ts)**2)
    
    step = tf.train.GradientDescentOptimizer(1.0).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_x = [[0,0], [0,1], [1,0], [1,1]]
        train_y = [ [0],   [1],   [1],   [1] ]
        
        train_errors = list()
        print('epoch', 'train error', 'W', 'b')
        for epoch in range(1, 500+1):
            s.run([ step ], { xs: train_x, ts: train_y })

            [ curr_W, curr_b ] = s.run([ W, b ], { })
            [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y })
            train_errors.append(train_error)
            
            if epoch%50 == 0:
                print(epoch, train_error, curr_W.tolist(), curr_b.tolist())
                
                (all_x0s, all_x1s) = np.meshgrid(np.linspace(0.0, 1.0, 50), np.linspace(0.0, 1.0, 50))
                [ all_ys ] = s.run([ ys ], { xs: np.stack([np.reshape(all_x0s, [-1]), np.reshape(all_x1s, [-1])], axis=1) })
                all_ys = np.reshape(all_ys, [50, 50])
                
                ax[0].cla()
                ax[0].contourf(all_x0s, all_x1s, all_ys, 100, vmin=0.0, vmax=1.0, cmap='bwr')
                ax[0].set_xlim(0.0, 1.0)
                ax[0].set_xlabel('x0')
                ax[0].set_ylim(0.0, 1.0)
                ax[0].set_ylabel('x1')
                ax[0].set_title('Logistic regression')
                ax[0].grid(True)
                
                ax[1].cla()
                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].set_xlim(-10, 500)
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
