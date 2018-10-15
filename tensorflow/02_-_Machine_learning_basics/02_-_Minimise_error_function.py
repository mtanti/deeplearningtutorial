import matplotlib.pyplot as plt
import tensorflow as tf

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

    error = (y1 - y2)**2
    [ grad ] = tf.gradients([ error ], [ min_x ])

    step = tf.assign(min_x, min_x - 0.01*grad)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        #Plot the quadratic equations
        inputs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        (fig, ax) = plt.subplots(1, 1)
        results_y1 = [ s.run([ y1 ], { min_x: i })[0] for i in inputs ]
        ax.plot(inputs, results_y1, color='red', linestyle='-', linewidth=3)
        results_y2 = [ s.run([ y2 ], { min_x: i })[0] for i in inputs ]
        ax.plot(inputs, results_y2, color='magenta', linestyle='-', linewidth=3)
        ax.set_xlim(-2.0, 2.0)
        ax.set_xlabel('x')
        ax.set_ylim(-10.0, 10.0)
        ax.set_ylabel('y')
        ax.grid(True)

        #Find where each new min_x lands on the graphs
        print('epoch', 'x', 'error')
        min_xs = list()
        min_ys = list()
        for epoch in range(1, 10+1):
            [ curr_x, curr_error ] = s.run([ min_x, error ], {})
            min_xs.append(curr_x)
            min_ys.append(0)
            print(epoch, curr_x, curr_error)

            s.run([ step ], {})
        ax.plot(min_xs, min_ys, color='blue', marker='o', markersize=10)
        fig.show()
