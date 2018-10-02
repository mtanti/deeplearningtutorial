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
    tf.summary.FileWriter('.', g)

    with tf.Session() as s:
        s.run([ init ], { })
        inputs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        results = [ s.run([ y ], { x: i })[0] for i in inputs ]

        plt.figure(1) #Optional: Figure ID
        plt.subplot(1, 1, 1) #Optional: Subplots rows, subplots columns, subplot position
        plt.title('Polynomial') #Optional: Set the title for the subplot
        plt.plot(inputs, results, linestyle='-', color='red', linewidth=3) #Plot the points in (inputs, results)
        plt.xlim(-2.0, 2.0) #Optional: Set the range for the x-axis
        plt.xlabel('x') #Optional: Set the label for the x-axis
        plt.ylim(-10.0, 10.0) #Optional: Set the range for the y-axis
        plt.ylabel('y') #Optional: Set the label for the y-axis
        plt.grid(True) #Optional: Show a grid

        plt.show()
