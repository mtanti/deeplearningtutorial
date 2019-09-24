import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

g = tf.Graph() #Create an empty graph.
with g.as_default(): #Set it as the default graph.
    #Add two float scalar input nodes to the graph.
    a = tf.placeholder(tf.float32, [], 'a') #Float scalar named 'a'.
    b = tf.placeholder(tf.float32, [], 'b') #Float scalar named 'b'.

    #Add an addition node to the graph.
    c = a + b

    #Make the graph read-only.
    g.finalize()

    #For TensorBoard visualisation (optional).
    tf.summary.FileWriter('.', g) #Save TensorBoard data in the current folder (you can change '.' to some other directory if you want).

    #Create a session to run the graph.
    with tf.Session() as s:
        [ result ] = s.run([ c ], { a: 1.0, b: 2.0 })
        print(result) #Result is of type numpy.float32.

#Graph and session close upon exit of the 'with' block.