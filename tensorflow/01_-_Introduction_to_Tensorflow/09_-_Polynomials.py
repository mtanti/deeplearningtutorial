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
        [ result ] = s.run([ y ], { x: 1 })
        print(result)
