import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

state_size = 2
embedding_size = 2
init_stddev = 0.01

class Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self):
        super().__init__()
        self.W = None
        self.b = None

    @property
    def state_size(self):
        return state_size

    @property
    def output_size(self):
        return state_size

    def build(self, input_shape):
        self.W = self.add_variable('W', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        self.b = self.add_variable('b', [state_size], tf.float32, tf.zeros_initializer())
        self.built = True

    def call(self, x, curr_state):
        layer_input = tf.concat([ curr_state, x ], axis=1)
        new_state = tf.tanh(tf.matmul(layer_input, self.W) + self.b)
        return (new_state, new_state) #Return the state as both output and state


g = tf.Graph()
with g.as_default():
    seqs = tf.constant(
        [
            [[1, 1], [2, 2], [3, 3]],
        ], tf.float32, [1, 3, 2], 'seqs'
    )
    
    seq_len = tf.constant(
        [
            2,
        ], tf.int32, [1]
    )
    
    init_state_fw = tf.constant([[0, 0]], tf.float32, [1, 2], 'init_state_fw')
    init_state_bw = tf.constant([[0, 0]], tf.float32, [1, 2], 'init_state_bw')

    cell_fw = Cell()
    cell_bw = Cell()
    
    ((outputs_fw, outputs_bw), (state_fw, state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seqs, sequence_length=seq_len, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
    
    #Note that you can use the same RNN for both forward and backward encodings by using just one cell variable
    
    #This is to encode the whole sequence
    state = tf.concat([ state_fw, state_bw ], axis=1)
    
    #This is to encode each element in the sequence
    outputs = tf.concat([ outputs_fw, outputs_bw ], axis=2)
    
    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })
        
        [
            curr_outputs_fw, curr_outputs_bw, curr_outputs,
            curr_state_fw, curr_state_bw, curr_state
        ] = s.run([
            outputs_fw, outputs_bw, outputs,
            state_fw, state_bw, state
        ], { })

        print('-------------')
        print('outputs_fw')
        print(curr_outputs_fw)
        print()
        
        print('-------------')
        print('outputs_bw')
        print(curr_outputs_bw)
        print()
        
        print('-------------')
        print('outputs')
        print(curr_outputs)
        print()
        
        print('-------------')
        print('state_fw')
        print(curr_state_fw)
        print()
        
        print('-------------')
        print('state_bw')
        print(curr_state_bw)
        print()
        
        print('-------------')
        print('state')
        print(curr_state)
        print()