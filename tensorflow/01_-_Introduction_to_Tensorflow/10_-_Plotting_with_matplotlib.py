import matplotlib.pyplot as plt
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, [], 'x')

    a = tf.get_variable('a', [], tf.float32, tf.random_normal_initializer())
    b = tf.get_variable('b', [], tf.float32, tf.random_normal_initializer())
    c = tf.get_variable('c', [], tf.float32, tf.random_normal_initializer())
    d = tf.get_variable('d', [], tf.float32, tf.random_normal_initializer())

    y = a + b*x + c*x**2 + d*x**3

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })
        inputs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        results = [ s.run([ y ], { x: i })[0] for i in inputs ]

        (fig, ax) = plt.subplots(1, 1) #Set the figure with the number of rows and columns of subplots (using more than one subplot will make 'ax' a list of each subplot's axis)
        ax.cla() #Optional: Clear the subplot
        ax.plot(inputs, results, color='red', linestyle='-', linewidth=3) #Plot the points in (inputs, results)
        ax.set_title('Polynomial') #Optional: Set the title for the subplot
        ax.set_xlim(-2.0, 2.0) #Optional: Set the range for the x-axis
        ax.set_xlabel('x') #Optional: Set the label for the x-axis
        ax.set_ylim(-10.0, 10.0) #Optional: Set the range for the y-axis
        ax.set_ylabel('y') #Optional: Set the label for the y-axis
        ax.grid(True) #Optional: Show a grid
        fig.tight_layout() #Optional: Make everything visible

        plt.show() #Show the figure
