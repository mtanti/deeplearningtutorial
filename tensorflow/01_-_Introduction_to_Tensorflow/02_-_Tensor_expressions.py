import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a = tf.placeholder(tf.float32, [3], 'a') #float vector of size 3

    one = tf.constant(1, tf.float32, [])
    two = tf.constant(2, tf.float32, [])
    b = two*a + one

    g.finalize()

    with tf.Session() as s:
        [ result ] = s.run([ b ], { a: [1.0, 2.0, 3.0] })
        print(result)
