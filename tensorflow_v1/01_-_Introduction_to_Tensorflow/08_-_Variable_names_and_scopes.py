import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

g = tf.Graph()
with g.as_default():
    with tf.variable_scope('a'): #Scope name is 'a'.
        tf.get_variable('v', [], tf.float32, tf.constant_initializer(1.0)) #Full variable name is 'a/v:0'.

    with tf.variable_scope('b'): #Scope name is 'b'.
        tf.get_variable('v', [], tf.float32, tf.constant_initializer(2.0)) #Full variable name is 'b/v:0'.

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        #Query graph for variable with given name.
        v = g.get_tensor_by_name('a/v:0')
        
        [ result ] = s.run([ v ], { })
        print(result)