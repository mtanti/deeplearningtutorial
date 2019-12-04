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
        return 2

    @property
    def output_size(self):
        return 2

    def build(self, input_shape):
        self.built = True

    def call(self, x, curr_state):
        new_state = x
        new_output = -x
        return (new_output, new_state)

###################################
class Model(object):
    
    def __init__(self):
        num_inputs = 1
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            init_state = tf.constant([ 0, 0 ], tf.float32, [2], 'init')

            batch_init_states = tf.tile(tf.reshape(init_state, [1, 2]), [ num_inputs, 1 ])

            seqs = tf.constant(
                    [
                        [ [1, 1], [2, 2], [3, 3] ],
                    ], tf.float32, [1, 3, 2], 'seqs'
                )

            seq_len = tf.constant(
                    [
                        2, #The length of the sequence is two input vectors long, with any other vectors after that to be ignored.
                    ], tf.float32, [1]
                )

            cell = Cell()
            (self.outputs, self.state) = tf.nn.dynamic_rnn(cell, seqs, sequence_length=seq_len, initial_state=batch_init_states)

            self.graph.finalize()

            self.sess = tf.Session()

    def close(self):
        self.sess.close()
    
    def output(self):
        return self.sess.run([ self.outputs, self.state ], {  })
    
###################################
model = Model()
[ outputs, state ] = model.output()
print('outputs:')
print(outputs)
print('state:', state)