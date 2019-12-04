import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

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
                        2,
                    ], tf.int32, [1]
                )
            
            #Simple RNN with a layer size of 2 and a tanh activation function.
            srnn_cell = tf.contrib.rnn.BasicRNNCell(2, tf.tanh)
            (self.srnn_outputs, self.srnn_state) = tf.nn.dynamic_rnn(srnn_cell, seqs, sequence_length=seq_len, initial_state=batch_init_states)
            
            #Gated Recurrent Unit with a layer size of 2.
            gru_cell = tf.contrib.rnn.GRUCell(2)
            (self.gru_outputs, self.gru_state) = tf.nn.dynamic_rnn(gru_cell, seqs, sequence_length=seq_len, initial_state=batch_init_states)
            
            '''
            GRU specification:
            def build(self, input_shape):
                self.W_g_z = self.add_variable('W_g_z', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.b_g_z = self.add_variable('b_g_z', [state_size], tf.float32, tf.zeros_initializer())
                self.W_g_r = self.add_variable('W_g_r', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.b_g_r = self.add_variable('b_g_r', [state_size], tf.float32, tf.zeros_initializer())
                self.W_s = self.add_variable('W_s', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.b_s = self.add_variable('b_s', [state_size], tf.float32, tf.zeros_initializer())
                self.built = True
            
            def call(self, x, curr_state):
                gate_input = tf.concat([ curr_state, x ], axis=1)
                g_z = tf.sigmoid(tf.matmul(gate_input, self.W_g_z) + self.b_g_z)
                g_r = tf.sigmoid(tf.matmul(gate_input, self.W_g_r) + self.b_g_r)
                state_input = tf.concat([ g_r*curr_state, x ], axis=1)
                new_state = g_z*tf.tanh(tf.matmul(state_input, self.W_s) + self.b_s) + (1 - g_z)*curr_state
                return (new_state, new_state)
            '''
            
            #Long short-term memory with a layer size of 2
            #LSTMs use two separate initial states which means that you need to initialise both states and also have two final states (although the 'h' state is what is usually used as a final state)
            lstm_cell = tf.nn.rnn_cell.LSTMCell(2)
            (self.lstm_outputs, lstm_state) = tf.nn.dynamic_rnn(lstm_cell, seqs, sequence_length=seq_len, initial_state=tf.contrib.rnn.LSTMStateTuple(h=batch_init_states, c=batch_init_states))
            self.lstm_state_h = lstm_state.h
            self.lstm_state_c = lstm_state.c
            
            '''
            LSTM specification:
            def build(self, input_shape):
                self.W_g_i = self.add_variable('W_g_i', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.b_g_i = self.add_variable('b_g_i', [state_size], tf.float32, tf.zeros_initializer())
                self.W_g_f = self.add_variable('W_g_f', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.b_g_f = self.add_variable('b_g_f', [state_size], tf.float32, tf.zeros_initializer())
                self.W_g_o = self.add_variable('W_g_o', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.b_g_o = self.add_variable('b_g_o', [state_size], tf.float32, tf.zeros_initializer())
                self.W_s = self.add_variable('W_s', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
                self.b_s = self.add_variable('b_s', [state_size], tf.float32, tf.zeros_initializer())
                self.built = True
            
            def call(self, x, curr_state):
                (curr_hidden, curr_cell) = curr_state
                gate_input = tf.concat([ curr_hidden, x ], axis=1)
                g_i = tf.sigmoid(tf.matmul(gate_input, self.W_g_i) + self.b_g_i)
                g_f = tf.sigmoid(tf.matmul(gate_input, self.W_g_f) + self.b_g_f)
                g_o = tf.sigmoid(tf.matmul(gate_input, self.W_g_o) + self.b_g_o)
                cell_input = tf.concat([ curr_hidden, x ], axis=1)
                new_cell = g_i*tf.tanh(tf.matmul(cell_input, self.W_s) + self.b_s) + g_f*curr_cell
                new_hidden = g_o*tf.tanh(new_cell)
                new_state = (new_hidden, new_cell)
                return (new_state, new_state)
            
            #The above implementation uses tuples instead of the LSTMTuple class, returns both hidden and cell states.
            #Don't forget to change the state_size and output_size to return (state_size, state_size).
            '''

            self.init = tf.global_variables_initializer()

            self.graph.finalize()

            self.sess = tf.Session()

    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def output(self):
        return self.sess.run([ self.srnn_outputs, self.srnn_state, self.gru_outputs, self.gru_state, self.lstm_outputs, self.lstm_state_h, self.lstm_state_c ], {  })
    
###################################
model = Model()
model.initialise()
[ srnn_outputs, srnn_state, gru_outputs, gru_state, lstm_outputs, lstm_state_h, lstm_state_c ] = model.output()
print('srnn_outputs:')
print(srnn_outputs)
print('srnn_state:')
print(srnn_state)
print()
print('gru_outputs')
print(gru_outputs)
print('gru_state')
print(gru_state)
print()
print('lstm_outputs:')
print(lstm_outputs)
print('lstm_state_h:')
print(lstm_state_h)
print('lstm_state_c:')
print(lstm_state_c)