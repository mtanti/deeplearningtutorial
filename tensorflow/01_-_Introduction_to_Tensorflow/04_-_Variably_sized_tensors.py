import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a = tf.placeholder(tf.float32, [None], 'a') #A vector of any size
    b = tf.placeholder(tf.float32, [None, 2], 'b') #A matrix with any number of rows but must have two columns

    g.finalize()

    with tf.Session() as s:
        [ result ] = s.run([ a ], { a: [1.0] })
        [ result ] = s.run([ a ], { a: [1.0, 2.0, 3.0] })
        [ result ] = s.run([ b ], { b: [[1.0, 2.0]] })
        [ result ] = s.run([ b ], { b: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] })