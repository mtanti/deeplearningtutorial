import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

#This function takes in a state and an input and transforms them into a new state: s(t+1) = s(t) + x(t)
def f(curr_state, x):
    print('curr_state:', curr_state)
    print('x:', x)
    print()
    new_state = curr_state + x
    return new_state

g = tf.Graph()
with g.as_default():
    init_state = tf.constant(0.0, tf.float32, [], 'init')

    y_f = f(init_state, tf.constant(1.0, tf.float32, []))

    seq = tf.constant(
        [ 1.0, 2.0, 3.0 ],
        tf.float32, [3], 'seq'
    )
    y_scan = tf.scan(f, seq, initializer=init_state)

    g.finalize()

    with tf.Session() as s:
        [ curr_y_f, curr_y_scan ] = s.run([ y_f, y_scan ], { })
        
        print('y_f:')
        print(curr_y_f)
        print()
        
        print('y_scan:')
        print(curr_y_scan)