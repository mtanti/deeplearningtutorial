import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

###################################
class Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, embed_size, state_size, init_stddev):
        super().__init__()
        self.W = None
        self.b = None
        self._embed_size = embed_size
        self._state_size = state_size
        self._init_stddev = init_stddev

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def build(self, input_shape):
        self.W = self.add_variable('W', [self._state_size+self._embed_size, self._state_size], tf.float32, tf.random_normal_initializer(stddev=self._init_stddev))
        self.b = self.add_variable('b', [self._state_size], tf.float32, tf.zeros_initializer())
        self.built = True

    def call(self, x, curr_state):
        layer_input = tf.concat([ curr_state, x ], axis=1)
        new_state = tf.tanh(tf.matmul(layer_input, self.W) + self.b)
        return (new_state, new_state) #Return the state as both output and state

###################################
class Model(object):
    
    def __init__(self):
        num_inputs = 1
        state_size = 2
        embed_size = 2
        init_stddev = 0.01
        
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
                        2,
                    ], tf.int32, [1]
                )
            
            init_state_fw = tf.constant([0, 0], tf.float32, [embed_size], 'init_state_fw')
            init_state_bw = tf.constant([0, 0], tf.float32, [embed_size], 'init_state_bw')
            
            batch_init_states_fw = tf.tile(tf.reshape(init_state_fw, [1, embed_size]), [ num_inputs, 1 ])
            batch_init_states_bw = tf.tile(tf.reshape(init_state_bw, [1, embed_size]), [ num_inputs, 1 ])

            cell_fw = Cell(embed_size, state_size, init_stddev)
            cell_bw = Cell(embed_size, state_size, init_stddev)
            
            ((self.outputs_fw, self.outputs_bw), (self.state_fw, self.state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seqs, sequence_length=seq_len, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
            
            #Note that you can use the same RNN for both forward and backward encodings by using just one cell variable.
            
            #This is to encode the whole sequence
            self.state = tf.concat([ self.state_fw, self.state_bw ], axis=1)
            
            #This is to encode each element in the sequence
            self.outputs = tf.concat([ self.outputs_fw, self.outputs_bw ], axis=2)

            self.init = tf.global_variables_initializer()

            self.graph.finalize()

            self.sess = tf.Session()

    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def output(self):
        return self.sess.run([ self.outputs_fw, self.outputs_bw, self.outputs, self.state_fw, self.state_bw, self.state ], {  })
    
###################################
model = Model()
model.initialise()
[ outputs_fw, outputs_bw, outputs, state_fw, state_bw, state ] = model.output()
print('outputs_fw:')
print(outputs_fw)
print('outputs_bw:')
print(outputs_bw)
print('outputs:')
print(outputs)
print()
print('state_fw:')
print(state_fw)
print('state_bw:')
print(state_bw)
print('state:')
print(state)
