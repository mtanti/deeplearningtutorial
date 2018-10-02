import tensorflow as tf

g = tf.Graph()
with g.as_default():
    with tf.variable_scope('a'):
        tf.get_variable('v', [], tf.float32)

    with tf.variable_scope('b'):
        tf.get_variable('v', [], tf.float32)

    init = tf.global_variables_initializer()
    
    g.finalize()
    tf.summary.FileWriter('.', g)

    with tf.Session() as s:
        s.run([ init ], {  })
        
        [ result ] = s.run([ g.get_tensor_by_name('a/v:0') ], { })
        print(result)
