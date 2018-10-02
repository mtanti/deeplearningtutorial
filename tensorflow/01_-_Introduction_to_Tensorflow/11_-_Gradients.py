import matplotlib.pyplot as plt
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, [], 'x')

    a = tf.get_variable('a', [], tf.float32, tf.constant_initializer(0))
    b = tf.get_variable('b', [], tf.float32, tf.constant_initializer(0))
    c = tf.get_variable('c', [], tf.float32, tf.constant_initializer(0))
    d = tf.get_variable('d', [], tf.float32, tf.constant_initializer(1))

    y = a + b*x + c*x**2 + d*x**3

    #Add nodes with the derivative of y with respect to x
    [ grad ] = tf.gradients([ y ], [ x ])

    init = tf.global_variables_initializer()

    g.finalize()
    tf.summary.FileWriter('.', g)

    with tf.Session() as s:
        s.run([ init ], { })
        inputs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        results_y = [ s.run([ y ], { x: i })[0] for i in inputs ]
        results_g = [ s.run([ grad ], { x: i })[0] for i in inputs ]

        plt.subplot(2, 1, 1)
        plt.title('Polynomial')
        plt.plot(inputs, results_y, linestyle='-', color='red', linewidth=3)
        plt.xlim(-2.0, 2.0)
        plt.ylim(-10.0, 10.0)
        plt.ylabel('y')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.title('Gradient')
        plt.plot(inputs, results_g, linestyle=':', color='blue', linewidth=3)
        plt.xlim(-2.0, 2.0)
        plt.xlabel('x')
        plt.ylim(-10.0, 10.0)
        plt.ylabel('y')
        plt.grid(True)

        plt.show()
