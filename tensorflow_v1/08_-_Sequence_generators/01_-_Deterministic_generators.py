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
    #Note: In this kind of sequence generator there cannot be a batch of inputs since it can only generate one sequence
    #This means that in the below code the batch size is assumed to be 1
    
    seq_len = tf.placeholder(tf.int32, [], 'seq_len') #The length of the sequence to generate
    target = tf.placeholder(tf.int32, [None], 'target') #The target sequence to generate during training
    
    input_vector = tf.get_variable('input_vector', [embedding_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev)) #The single input vector that is learned
    input_seq = tf.tile(tf.reshape(input_vector, [1, 1, embedding_size]), [1, seq_len, 1]) #Replicate the input vector for every item to generate in the sequence
    
    init_state = tf.get_variable('init_state', [state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    batch_init = tf.reshape(init_state, [1, state_size]) 
    
    cell = tf.contrib.rnn.BasicRNNCell(state_size, tf.tanh)
    (outputs, _) = tf.nn.dynamic_rnn(cell, input_seq, initial_state=batch_init)

    W = tf.get_variable('W', [state_size, vocab_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
    b = tf.get_variable('b', [vocab_size], tf.float32, tf.zeros_initializer())
    outputs_2d = tf.reshape(outputs, [seq_len, state_size]) #Flatten RNN outputs since there is only 1 batch item
    logits = tf.matmul(outputs_2d, W) + b #Note that matmul can only take 2D matrices so we always have to somehow reshape higher dimensional tensors before applying matmul
    probs = tf.nn.softmax(logits)

    error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits))
    
    step = tf.train.AdamOptimizer().minimize(error)

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
        
        [ curr_probs ] = s.run([ probs ], { seq_len: max_seq_len })
        print('Generated sequence')
        print(np.argmax(curr_probs, axis=1))
        
        fig.show()