import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None], 'xs') #The 'x' values of the points
    ts = tf.placeholder(tf.float32, [None], 'ts') #The 'y' values of the points (called targets)
    
    c0 = tf.get_variable('c0', [], tf.float32, tf.constant_initializer(0))
    c1 = tf.get_variable('c1', [], tf.float32, tf.constant_initializer(0))
    c2 = tf.get_variable('c2', [], tf.float32, tf.constant_initializer(0))
    c3 = tf.get_variable('c3', [], tf.float32, tf.constant_initializer(0))
    c4 = tf.get_variable('c4', [], tf.float32, tf.constant_initializer(0))
    c5 = tf.get_variable('c5', [], tf.float32, tf.constant_initializer(0))
    c6 = tf.get_variable('c6', [], tf.float32, tf.constant_initializer(0))
    ys = c0 + c1*xs + c2*xs**2 + c3*xs**3 + c4*xs**4 + c5*xs**5 + c6*xs**6 #The predicted 'y' values
    
    error = tf.reduce_mean((ys - ts)**2)

    step = tf.train.GradientDescentOptimizer(0.0005).minimize(error)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        points_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
        points_y = [3.22, 1.64, 0.58, 1.25, 5.07]
        
        print('epoch', 'error', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6')
        for epoch in range(1, 2000+1):
            [ curr_c0, curr_c1, curr_c2, curr_c3, curr_c4, curr_c5, curr_c6 ] = s.run([ c0, c1, c2, c3, c4, c5, c6 ], { })
            [ curr_error ] = s.run([ error ], { xs: points_x, ts: points_y })
            if epoch%10 == 0:
                print(epoch, curr_error, curr_c0, curr_c1, curr_c2, curr_c3, curr_c4, curr_c5, curr_c6)

            s.run([ step ], { xs: points_x, ts: points_y })

        all_xs = np.linspace(-2.5, 2.5, 50)
        [ all_ys ] = s.run([ ys ], {xs: all_xs})
        
        (fig, ax) = plt.subplots(1, 1)
        ax.plot(points_x, points_y, color='red', linestyle='', marker='o', markersize=10)
        ax.plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
        ax.set_xlim(-2.5, 2.5)
        ax.set_xlabel('x')
        ax.set_ylim(-10.0, 10.0)
        ax.set_ylabel('y')
        ax.grid(True)
        
        fig.tight_layout()
        fig.show()
