import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.0005
max_epochs = 2000

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None], 'xs') #The x values of the points
    ts = tf.placeholder(tf.float32, [None], 'ts') #The y values of the points (called targets)
    
    c0 = tf.get_variable('c0', [], tf.float32, tf.zeros_initializer())
    c1 = tf.get_variable('c1', [], tf.float32, tf.zeros_initializer())
    c2 = tf.get_variable('c2', [], tf.float32, tf.zeros_initializer())
    c3 = tf.get_variable('c3', [], tf.float32, tf.zeros_initializer())
    c4 = tf.get_variable('c4', [], tf.float32, tf.zeros_initializer())
    c5 = tf.get_variable('c5', [], tf.float32, tf.zeros_initializer())
    c6 = tf.get_variable('c6', [], tf.float32, tf.zeros_initializer())
    ys = c0 + c1*xs + c2*xs**2 + c3*xs**3 + c4*xs**4 + c5*xs**5 + c6*xs**6 #The predicted 'y' values
    
    error = tf.reduce_mean((ys - ts)**2)

    #Using Tensorflow's provided gradient descent function
    step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        #The training set of points to interpolate
        points_x = [ -2.0, -1.0, 0.0, 1.0, 2.0 ]
        points_y = [ 3.22, 1.64, 0.58, 1.25, 5.07 ]

        (fig, ax) = plt.subplots(1, 1)
        plt.ion() #Enable interactive plots in order to see the polynomial evolving
        
        ax.plot(points_x, points_y, color='red', linestyle='', marker='o', markersize=10)
        ax.set_xlim(-2.5, 2.5)
        ax.set_xlabel('x')
        ax.set_ylim(-10.0, 10.0)
        ax.set_ylabel('y')
        ax.grid(True)
        fig.tight_layout()
        
        all_xs = np.linspace(-2.5, 2.5, 30)
        [ all_ys ] = s.run([ ys ], { xs: all_xs })
        ax.plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
        plt.draw() #Draw the first polynomial before training has started in blue
        plt.pause(0.0001) #Wait for it to be displayed

        print('epoch', 'error', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', sep='\t')
        for epoch in range(1, max_epochs+1):
            [ curr_c0, curr_c1, curr_c2, curr_c3, curr_c4, curr_c5, curr_c6 ] = s.run([ c0, c1, c2, c3, c4, c5, c6 ], { }) #Get all coefficients at in one go
            [ curr_error ] = s.run([ error ], { xs: points_x, ts: points_y })
            if epoch%100 == 0: #Avoid printing information about every single epoch by only printing once every 100 epochs
                print(epoch, curr_error, round(curr_c0, 3), round(curr_c1, 3), round(curr_c2, 3), round(curr_c3, 3), round(curr_c4, 3), round(curr_c5, 3), round(curr_c6, 3), sep='\t')
                
                [ all_ys ] = s.run([ ys ], { xs: all_xs })
                ax.plot(all_xs, all_ys, color='magenta', linestyle='-', linewidth=1)
                plt.draw() #Draw the current polynomial found
                plt.pause(0.0001)

            s.run([ step ], { xs: points_x, ts: points_y })

        ax.plot(all_xs, all_ys, color='red', linestyle='-', linewidth=3) #Plot the final polynomial found in red

    fig.show()