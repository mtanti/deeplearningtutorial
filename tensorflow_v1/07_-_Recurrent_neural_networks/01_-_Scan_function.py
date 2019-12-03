import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

###################################
def f(curr_state, x):
    #This function takes in a state and an input and transforms them into a new state: s(t+1) = s(t) + x(t).
    print('curr_state:', curr_state)
    print('x:', x)
    print()
    new_state = curr_state + x
    return new_state

###################################
class Model(object):
    
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            init_state = tf.constant(0.0, tf.float32, [], 'init')

            print('f()')
            self.f_output = f(init_state, tf.constant(1.0, tf.float32, []))

            seq = tf.constant(
                [ 1.0, 2.0, 3.0 ],
                tf.float32, [3], 'seq'
            )
            
            print('tf.scan()')
            self.scan_output = tf.scan(f, seq, initializer=init_state)
            self.scan_last_output = self.scan_output[-1]

            self.graph.finalize()

            self.sess = tf.Session()

    def close(self):
        self.sess.close()
    
    def output(self):
        return self.sess.run([ self.f_output, self.scan_output, self.scan_last_output ], {  })
    
###################################
model = Model()
[ f_output, scan_output, scan_last_output ] = model.output()
print('f_output:', f_output)
print('scan_output:', scan_output)
print('scan_last_output:', scan_last_output)