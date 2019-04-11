import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.0001
max_epochs = 4500

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None], 'xs')
    ts = tf.placeholder(tf.float32, [None], 'ts')

    #Initialise the variables to small random values
    c0 = tf.get_variable('c0', [], tf.float32, tf.zeros_initializer())
    c1 = tf.get_variable('c1', [], tf.float32, tf.random_normal_initializer(mean=0.0, stddev=0.01)) #A random number from the normal distribution with a mean of zero and a standard deviation of 0.01
    c2 = tf.get_variable('c2', [], tf.float32, tf.random_normal_initializer(mean=0.0, stddev=0.01))
    c3 = tf.get_variable('c3', [], tf.float32, tf.random_normal_initializer(mean=0.0, stddev=0.01))
    c4 = tf.get_variable('c4', [], tf.float32, tf.random_normal_initializer(mean=0.0, stddev=0.01))
    c5 = tf.get_variable('c5', [], tf.float32, tf.random_normal_initializer(mean=0.0, stddev=0.01))
    c6 = tf.get_variable('c6', [], tf.float32, tf.random_normal_initializer(mean=0.0, stddev=0.01))
    #Note that there is no need to specify a mean of 0.0 as this is the default, so it can just be left out
    
    ys = c0 + c1*xs + c2*xs**2 + c3*xs**3 + c4*xs**4 + c5*xs**5 + c6*xs**6
    
    error = tf.reduce_mean((ys - ts)**2)
    
    step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 2)
        plt.ion()
        
        train_x = [ -2.0, -1.0, 0.0, 1.0, 2.0 ]
        train_y = [ 3.22, 1.64, 0.58, 1.25, 5.07 ]
        val_x   = [  -1.75, -0.75, 0.25, 1.25 ]
        val_y   = [ 3.03, 0.64, 0.46, 0.77 ]
        test_x  = [ -1.5, -0.5, 0.5, 1.5 ]
        test_y  = [ 2.38, 0.05, 0.47, 1.67 ]
        
        train_errors = list()
        val_errors = list()
        print('epoch', 'trainerror', 'valerror', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', sep='\t')
        for epoch in range(1, max_epochs+1):
            s.run([ step ], { xs: train_x, ts: train_y })

            [ curr_c0, curr_c1, curr_c2, curr_c3, curr_c4, curr_c5, curr_c6 ] = s.run([ c0, c1, c2, c3, c4, c5, c6 ], { })
            [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y })
            [ val_error ]  = s.run([ error ], { xs: val_x,  ts: val_y })
            train_errors.append(train_error)
            val_errors.append(val_error)

            if epoch%100 == 0:
                print(epoch, train_error, val_error, round(curr_c0, 3), round(curr_c1, 3), round(curr_c2, 3), round(curr_c3, 3), round(curr_c4, 3), round(curr_c5, 3), round(curr_c6, 3), sep='\t')
                
                ax[0].cla()
                ax[1].cla()

                all_xs = np.linspace(-2.5, 2.5, 30)
                [ all_ys ] = s.run([ ys ], { xs: all_xs })
                ax[0].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
                ax[0].plot(train_x, train_y, color='red', linestyle='', marker='o', markersize=10, label='train')
                ax[0].plot(val_x, val_y, color='yellow', linestyle='', marker='o', markersize=10, label='val')
                ax[0].plot(test_x, test_y, color='orange', linestyle='', marker='o', markersize=10, label='test')
                ax[0].set_xlim(-2.5, 2.5)
                ax[0].set_xlabel('x')
                ax[0].set_ylim(-10.0, 10.0)
                ax[0].set_ylabel('y')
                ax[0].set_title('Polynomial')
                ax[0].grid(True)
                ax[0].legend()

                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].plot(np.arange(len(val_errors)), val_errors, color='yellow', linestyle='-', label='val')
                ax[1].set_xlim(0, max_epochs)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0, 1)
                ax[1].set_ylabel('MSE')
                ax[1].grid(True)
                ax[1].set_title('Error progress')
                ax[1].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        [ test_error ]  = s.run([ error ], { xs: test_x,  ts: test_y })
        ax[1].annotate('Test error: '+str(test_error), (0,0))
        print()
        print('Test error:', test_error)
        
        fig.show()