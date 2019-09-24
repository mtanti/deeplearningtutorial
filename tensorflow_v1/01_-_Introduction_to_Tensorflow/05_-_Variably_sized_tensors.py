import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

g = tf.Graph()
with g.as_default():
    a = tf.placeholder(tf.float32, [None], 'a') #A vector of any size.
    b = tf.placeholder(tf.float32, [None, 2], 'b') #A matrix with any number of rows but must have two columns.
    
    #This will tell you the complete shape at run time.
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)

    g.finalize()

    with tf.Session() as s:
        [ result ] = s.run([ a_shape ], { a: [1.0] })
        print(result)
        
        [ result ] = s.run([ a_shape ], { a: [1.0, 2.0, 3.0] })
        print(result)
        
        [ result ] = s.run([ b_shape ], { b: [[1.0, 2.0]] })
        print(result)
        
        [ result ] = s.run([ b_shape ], { b: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] })
        print(result)