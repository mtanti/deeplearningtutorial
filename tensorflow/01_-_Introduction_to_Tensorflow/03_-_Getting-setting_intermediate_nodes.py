import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a = tf.placeholder(tf.float32, [3], 'a')

    b = 2*a
    c = b + 1
    d = 2*c

    g.finalize()

    with tf.Session() as s:
        [ result ] = s.run([ c ], { b: [1.0, 2.0, 3.0] })
        print(result)
