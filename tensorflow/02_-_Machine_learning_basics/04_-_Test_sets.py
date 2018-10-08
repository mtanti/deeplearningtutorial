import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None], 'xs')
    ts = tf.placeholder(tf.float32, [None], 'ts')
    
    c0 = tf.get_variable('c0', [], tf.float32, tf.constant_initializer(0))
    c1 = tf.get_variable('c1', [], tf.float32, tf.constant_initializer(0))
    c2 = tf.get_variable('c2', [], tf.float32, tf.constant_initializer(0))
    c3 = tf.get_variable('c3', [], tf.float32, tf.constant_initializer(0))
    c4 = tf.get_variable('c4', [], tf.float32, tf.constant_initializer(0))
    c5 = tf.get_variable('c5', [], tf.float32, tf.constant_initializer(0))
    c6 = tf.get_variable('c6', [], tf.float32, tf.constant_initializer(0))
    ys = c0 + c1*xs + c2*xs**2 + c3*xs**3 + c4*xs**4 + c5*xs**5 + c6*xs**6
    
    error = tf.reduce_mean((ys - ts)**2)

    step = tf.train.GradientDescentOptimizer(0.0005).minimize(error)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        #Create matplotlib figure with 2 subplots
        (fig, ax) = plt.subplots(1, 2)
        plt.ion() #Make figure interactive so that it can be updated during training
        
        train_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
        train_y = [3.22, 1.64, 0.58, 1.25, 5.07]
        test_x  = [-1.5, -0.5, 0.5, 1.5]
        test_y  = [2.38, 0.05, 0.47, 1.67]
        
        train_errors = list()
        test_errors = list()
        print('epoch', 'trainerror', 'testerror', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6')
        for epoch in range(2000):
            s.run([ step ], { xs: train_x, ts: train_y })

            [ curr_c0, curr_c1, curr_c2, curr_c3, curr_c4, curr_c5, curr_c6 ] = s.run([ c0, c1, c2, c3, c4, c5, c6 ], { })
            [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y })
            [ test_error ]  = s.run([ error ], { xs: test_x,  ts: test_y })
            train_errors.append(train_error)
            test_errors.append(test_error)

            if epoch%50 == 0: #Update figure every 50th epoch
                print(epoch, train_error, test_error, curr_c0, curr_c1, curr_c2, curr_c3, curr_c4, curr_c5, curr_c6)
                
                #Clear subplots
                ax[0].cla()
                ax[1].cla()

                #Plot points
                ax[0].plot(train_x, train_y, color='red', linestyle='', marker='o', markersize=10, label='train')
                ax[0].plot(test_x, test_y, color='orange', linestyle='', marker='o', markersize=10, label='test')
                ax[0].set_xlim(-2.5, 2.5)
                ax[0].set_xlabel('x')
                ax[0].set_ylim(-10.0, 10.0)
                ax[0].set_ylabel('y')
                ax[0].set_title('Polynomial')
                ax[0].grid(True)
                ax[0].legend()

                #Plot polynomial
                all_xs = np.arange(-2.5, 2.5+0.1, 0.1)
                [ all_ys ] = s.run([ ys ], {xs: all_xs})
                ax[0].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)

                #Plot error progress
                ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax[1].plot(np.arange(len(test_errors)), test_errors, color='orange', linestyle='-', label='test')
                ax[1].set_xlim(-10, 2000)
                ax[1].set_xlabel('epoch')
                ax[1].set_ylim(0, 1)
                ax[1].set_ylabel('MSE')
                ax[1].grid(True)
                ax[1].set_title('Error progress')
                ax[1].legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)
        
        #Show test error at the end
        [ test_error ]  = s.run([ error ], { xs: test_x,  ts: test_y })
        ax[1].annotate('Test error: '+str(test_error), (0,0))
        fig.show()
