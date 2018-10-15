import matplotlib.pyplot as plt
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    min_x = tf.get_variable('min_x', [], tf.float32, tf.constant_initializer(1))

    a = tf.get_variable('a', [], tf.float32, tf.constant_initializer(0))
    b = tf.get_variable('b', [], tf.float32, tf.constant_initializer(0))
    c = tf.get_variable('c', [], tf.float32, tf.constant_initializer(1))
    d = tf.get_variable('d', [], tf.float32, tf.constant_initializer(0))

    y = a + b*min_x + c*min_x**2 + d*min_x**3
    
    [ grad ] = tf.gradients([ y ], [ min_x ])

    step = tf.assign(min_x, min_x - 0.2*grad) #Gradient descent equation with a learning rate of 0.2

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        #Plot the quadratic equation
        inputs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        results_y = [ s.run([ y ], { min_x: i })[0] for i in inputs ]
        (fig, ax) = plt.subplots(1, 1)
        ax.plot(inputs, results_y, color='red', linestyle='-', linewidth=3)
        ax.set_xlim(-2.0, 2.0)
        ax.set_xlabel('x')
        ax.set_ylim(-10.0, 10.0)
        ax.set_ylabel('y')
        ax.grid(True)

        #Find where each new min_x lands on the graph
        print('epoch', 'x', 'y')
        min_xs = list()
        min_ys = list()
        for epoch in range(1, 10+1): #Optimize min_x for 10 times (epochs)
            [ curr_x, curr_y ] = s.run([ min_x, y ], {})
            min_xs.append(curr_x)
            min_ys.append(curr_y)
            print(epoch, curr_x, curr_y)

            #Optimize min_x a little here
            s.run([ step ], {})
        ax.plot(min_xs, min_ys, color='blue', marker='o', markersize=10)
        fig.show()
