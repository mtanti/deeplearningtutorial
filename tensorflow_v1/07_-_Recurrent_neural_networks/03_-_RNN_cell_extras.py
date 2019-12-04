import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

###################################
class Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self):
        super().__init__()

    @property
    def state_size(self):
        return (2, 2) #Vector size of each vector being returned.

    @property
    def output_size(self):
        return (2, 2) #Vector size of each vector being returned.

    def build(self, input_shape):
        self.built = True

    def call(self, x, curr_state):
        (curr_state1, curr_state2) = curr_state

        new_state1 = curr_state1 + x
        new_output1 = new_state1
        
        new_state2 = curr_state2 + new_state1
        new_output2 = new_state2
        
        return ((new_output1, new_output2), (new_state1, new_state2)) #Return all the vectors in a tuple.

###################################
class Model(object):
    
    def __init__(self):
        num_inputs = 1
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            #Separate initial states for each vector.
            init_state1 = tf.constant([ 0, 0 ], tf.float32, [2], 'init')
            init_state2 = tf.constant([ 0, 0 ], tf.float32, [2], 'init')

            batch_init_states1 = tf.tile(tf.reshape(init_state1, [1, 2]), [ num_inputs, 1 ])
            batch_init_states2 = tf.tile(tf.reshape(init_state2, [1, 2]), [ num_inputs, 1 ])

            batch_init_states = (batch_init_states1, batch_init_states2) #Tuple of initial states.

            seqs = tf.constant(
                    [
                        [ [1, 1], [2, 2], [3, 3] ],
                    ], tf.float32, [1, 3, 2], 'seqs'
                )
            
            cell = Cell()
            ((self.outputs1, self.outputs2), (self.state1, self.state2)) = tf.nn.dynamic_rnn(cell, seqs, initial_state=batch_init_states)

            self.graph.finalize()

            self.sess = tf.Session()

    def close(self):
        self.sess.close()
    
    def output(self):
        return self.sess.run([ self.outputs1, self.outputs2, self.state1, self.state2 ], {  })
    
###################################
model = Model()
[ outputs1, outputs2, state1, state2 ] = model.output()
print('outputs1:')
print(outputs1)
print('outputs2:')
print(outputs2)
print('state1:', state1)
print('state2:', state2)