import tensorflow as tf

g = tf.Graph()
with g.as_default():
    v = tf.get_variable('v', [], tf.float32)

    v_in = tf.placeholder(tf.float32, [], 'v_in')
    v_setter = tf.assign(v, v_in) #Run this to set v to v_in
    
    g.finalize()

    with tf.Session() as s:
        s.run([ v_setter ], { v_in: 1.0 })
        
        [ result ] = s.run([ v ], { })
        print(result)

        s.run([ v_setter ], { v_in: 2.0 })

        [ result ] = s.run([ v ], { })
        print(result)
