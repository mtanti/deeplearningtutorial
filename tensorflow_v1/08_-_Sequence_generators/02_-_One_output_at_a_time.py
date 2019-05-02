import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

max_epochs = 2000
init_stddev = 0.001
embedding_size = 2
state_size = 2
vocab_size = 2
max_seq_len = 10

training_sequence = [ 0, 1, 0, 1, 0, 1 ]

g = tf.Graph()
with g.as_default():
    seq_len = tf.placeholder(tf.int32, [], 'seq_len')
    target = tf.placeholder(tf.int32, [None], 'target')
    
    input_vector = tf.get_variable('input_vector', [embedding_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    input_seq = tf.tile(tf.reshape(input_vector, [1, 1, embedding_size]), [1, seq_len, 1])
    
    init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    batch_init = tf.reshape(init_state, [1, state_size]) 
    
    cell = tf.contrib.rnn.BasicRNNCell(state_size, tf.tanh)
    (outputs, _) = tf.nn.dynamic_rnn(cell, input_seq, initial_state=batch_init)

    W = tf.get_variable('W', [state_size, vocab_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    b = tf.get_variable('b', [vocab_size], tf.float32, tf.zeros_initializer())
    outputs_2d = tf.reshape(outputs, [seq_len, state_size])
    logits = tf.matmul(outputs_2d, W) + b
    probs = tf.nn.softmax(logits)

    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits))
    
    step = tf.train.AdamOptimizer().minimize(error)

    #This is a separate part of the neural network that is only used at test time and only uses variables that are already defined above
    onestep_state = tf.placeholder(tf.float32, [None, state_size], 'onestep_state') #The state needs to be supplied externally
    onestep_batch_input = tf.reshape(input_vector, [1, embedding_size])
    (onestep_new_state, _) = cell.call(onestep_batch_input, onestep_state) #The supplied state is passed to a single time step call of the RNN
    onestep_probs = tf.nn.softmax(tf.matmul(onestep_new_state, W) + b)

    init = tf.global_variables_initializer()

    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })

        (fig, ax) = plt.subplots(1, 1)
        plt.ion()
        
        train_errors = list()
        print('epoch', 'train error', sep='\t')
        for epoch in range(1, max_epochs+1):
            s.run([ step ], { seq_len: len(training_sequence), target: training_sequence })

            [ train_error ] = s.run([ error ], { seq_len: len(training_sequence), target: training_sequence })
            train_errors.append(train_error)
            
            if epoch%100 == 0:
                print(epoch, train_error, sep='\t')

                ax.cla()
                ax.plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                ax.set_xlim(0, max_epochs)
                ax.set_xlabel('epoch')
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel('XE') #Cross entropy
                ax.grid(True)
                ax.set_title('Error progress')
                ax.legend()
                
                fig.tight_layout()
                plt.draw()
                plt.pause(0.0001)

        print()

        print('Generated sequence')
        [ state ] = s.run([ init_state ], { })
        state = np.array([ state ]) #Make a batch
        for _ in range(max_seq_len):
            [ curr_probs, state ] = s.run([ onestep_probs, onestep_new_state ], { onestep_state: state })
            print(np.argmax(curr_probs), end=' ')
        
        fig.show()