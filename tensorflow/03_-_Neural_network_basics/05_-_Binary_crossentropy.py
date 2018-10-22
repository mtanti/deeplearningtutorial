import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None], 'xs')
    
    ys = tf.sigmoid(xs)

    square_error = (ys - 1)**2
    binary_cross_entropy = -tf.log(ys) #Can also be tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(xs), logits=xs)
    
    #Note that sigmoid_cross_entropy_with_logits takes the input to the sigmoid (called the logits) rather than the sigmoid itself in order to avoid numerical overflows.
    
    #The above error functions are used to make the sigmoid output a 1. The following is to make it output a 0:
    #square_error = (ys - 0)**2
    #binary_cross_entropy = -tf.log(1 - ys) #Can also be tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(xs), logits=xs)
    
    g.finalize()

    with tf.Session() as s:
        (fig, ax) = plt.subplots(1, 2)
        
        all_xs = np.linspace(-10, 10, 50)
        
        [ all_ys ] = s.run([ ys ], {xs: all_xs})
        ax[0].plot(all_xs, all_ys, color='red', linestyle=':', linewidth=3, label='sigmoid')
        [ all_ys ] = s.run([ square_error ], {xs: all_xs})
        ax[0].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3, label='error')
        ax[0].set_xlim(-10, 10)
        ax[0].set_xlabel('x')
        ax[0].set_ylim(-0.1, 5.0)
        ax[0].set_ylabel('y')
        ax[0].set_title('Square error (make y=1)')
        ax[0].legend()
        ax[0].grid(True)

        [ all_ys ] = s.run([ ys ], {xs: all_xs})
        ax[1].plot(all_xs, all_ys, color='red', linestyle=':', linewidth=3, label='sigmoid')
        [ all_ys ] = s.run([ binary_cross_entropy ], {xs: all_xs})
        ax[1].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3, label='error')
        ax[1].set_xlim(-10, 10)
        ax[1].set_xlabel('x')
        ax[1].set_ylim(-0.1, 5.0)
        ax[1].set_title('Binary cross entropy (make y=1)')
        ax[1].legend()
        ax[1].grid(True)

        fig.tight_layout()
        fig.show()
