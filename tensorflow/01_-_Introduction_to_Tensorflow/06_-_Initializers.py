import tensorflow as tf

g = tf.Graph()
with g.as_default():
    #Initialise with zeros
    v1 = tf.get_variable('v1', [], tf.float32, tf.zeros_initializer())
    #Initialise with a constant
    v2 = tf.get_variable('v2', [], tf.float32, tf.constant_initializer(1.0))
    #Initialise randomly using a normal distribution
    v3 = tf.get_variable('v3', [], tf.float32, tf.random_normal_initializer ())

    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })
        
        [ result ] = s.run([ v1 ], { })
        print(result)

        [ result ] = s.run([ v2 ], { })
        print(result)

        [ result ] = s.run([ v3 ], { })
        print(result)
