import matplotlib.pyplot as plt
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, [], 'x')

    a = tf.get_variable('a', [], tf.float32, tf.constant_initializer(0))
    b = tf.get_variable('b', [], tf.float32, tf.constant_initializer(0))
    c = tf.get_variable('c', [], tf.float32, tf.constant_initializer(0))
    d = tf.get_variable('d', [], tf.float32, tf.constant_initializer(1))

    y = a + b*x + c*x**2 + d*x**3

    #Add nodes to graph that calculate the derivative of y with respect to x
    [ grad ] = tf.gradients([ y ], [ x ])

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })
        
        coefficients = s.run([ a, b, c, d ], { })
        print(coefficients)
        
        x_values = [ -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0 ]
        y_values = [ s.run([ y ], { x: val })[0] for val in x_values ]
        g_values = [ s.run([ grad ], { x: val })[0] for val in x_values ]

        (fig, ax) = plt.subplots(2, 1) #A figure with two subplots under each other
        ax[0].set_title('Polynomial')
        ax[0].plot(x_values, y_values, color='red', linestyle='-', linewidth=3)
        ax[0].set_xlim(-2.0, 2.0)
        ax[0].set_ylim(-10.0, 10.0)
        ax[0].set_ylabel('y')
        ax[0].grid(True)
        
        ax[1].set_title('Gradient')
        ax[1].plot(x_values, g_values, color='blue', linestyle=':', linewidth=3)
        ax[1].set_xlim(-2.0, 2.0)
        ax[1].set_xlabel('x')
        ax[1].set_ylim(-10.0, 10.0)
        ax[1].set_ylabel('dy/dx')
        ax[1].grid(True)
        
        fig.tight_layout()
        plt.show()
