import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.01
max_epochs = 10

g = tf.Graph()
with g.as_default():
    min_x = tf.get_variable('min_x', [], tf.float32, tf.constant_initializer(1))

    a1 = tf.get_variable('a1', [], tf.float32, tf.constant_initializer(-2))
    b1 = tf.get_variable('b1', [], tf.float32, tf.constant_initializer(-2))
    c1 = tf.get_variable('c1', [], tf.float32, tf.constant_initializer(1))
    d1 = tf.get_variable('d1', [], tf.float32, tf.constant_initializer(0))
    y1 = a1 + b1*min_x + c1*min_x**2 + d1*min_x**3

    a2 = tf.get_variable('a2', [], tf.float32, tf.constant_initializer(-1))
    b2 = tf.get_variable('b2', [], tf.float32, tf.constant_initializer(2))
    c2 = tf.get_variable('c2', [], tf.float32, tf.constant_initializer(1))
    d2 = tf.get_variable('d2', [], tf.float32, tf.constant_initializer(0))
    y2 = a2 + b2*min_x + c2*min_x**2 + d2*min_x**3

    #The square error function
    error = (y1 - y2)**2

    [ grad ] = tf.gradients([ error ], [ min_x ])

    step = tf.assign(min_x, min_x - learning_rate*grad)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        inputs = np.linspace(-2.0, 2.0, 20)
        (fig, ax) = plt.subplots(2, 1)
        
        #Plot the quadratic equations
        results_y1 = [ s.run([ y1 ], { min_x: i })[0] for i in inputs ]
        ax[0].plot(inputs, results_y1, color='red', linestyle='-', linewidth=3)
        results_y2 = [ s.run([ y2 ], { min_x: i })[0] for i in inputs ]
        ax[0].plot(inputs, results_y2, color='magenta', linestyle='-', linewidth=3)
        ax[0].set_xlim(-2.0, 2.0)
        ax[0].set_xlabel('x')
        ax[0].set_ylim(-10.0, 10.0)
        ax[0].set_ylabel('y')
        ax[0].grid(True)

        #Plot the error function
        results_error = [ s.run([ error ], { min_x: i })[0] for i in inputs ]
        ax[1].plot(inputs, results_error, color='blue', linestyle='-', linewidth=3)
        ax[1].set_xlim(-2.0, 2.0)
        ax[1].set_xlabel('x')
        ax[1].set_ylim(-1.0, 50.0)
        ax[1].set_ylabel('y')
        ax[1].grid(True)
        
        fig.tight_layout()

        #Find where each new min_x lands on the graphs
        print('epoch', 'x', 'error', sep='\t')
        min_xs = list()
        min_ys = list()
        for epoch in range(1, max_epochs+1):
            [ curr_x, curr_error ] = s.run([ min_x, error ], {})
            min_xs.append(curr_x)
            min_ys.append(curr_error)
            print(epoch, curr_x, curr_error, sep='\t')

            s.run([ step ], {})

        ax[1].plot(min_xs, min_ys, color='blue', marker='o', markersize=10)
        fig.show()
