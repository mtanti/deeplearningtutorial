import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, [], 'x')

    #Co-efficients of a polynomial of degree 3 (y = a + bx + cx^2 + dx^3).
    a = tf.get_variable('a', [], tf.float32, tf.random_normal_initializer())
    b = tf.get_variable('b', [], tf.float32, tf.random_normal_initializer())
    c = tf.get_variable('c', [], tf.float32, tf.random_normal_initializer())
    d = tf.get_variable('d', [], tf.float32, tf.random_normal_initializer())

    #The polynomial.
    y = a + b*x + c*x**2 + d*x**3

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })
        
        coefficients = s.run([ a, b, c, d ], { })
        print('coefficients:', coefficients)
        print()
        
        [ result ] = s.run([ y ], { x: -1.0 })
        print('f(-1) =', result)
        
        [ result ] = s.run([ y ], { x: 0.0 })
        print('f( 0) =', result)
        
        [ result ] = s.run([ y ], { x: 1.0 })
        print('f( 1) =', result)