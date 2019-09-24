import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

g = tf.Graph()
with g.as_default():
    v = tf.get_variable('v', [], tf.float32)

    v_in = tf.placeholder(tf.float32, [], 'v_in')
    v_setter = tf.assign(v, v_in) #Run this to set the value in v_in to the variable v.
    
    g.finalize()

    with tf.Session() as s:
        #Set value of v to 1.0.
        s.run([ v_setter ], { v_in: 1.0 })
        
        [ result ] = s.run([ v ], { })
        print(result)

        #Change value of v to 2.0.
        s.run([ v_setter ], { v_in: 2.0 })

        [ result ] = s.run([ v ], { })
        print(result)