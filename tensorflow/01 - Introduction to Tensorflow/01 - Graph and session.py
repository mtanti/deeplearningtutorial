import tensorflow as tf

g = tf.Graph() #Create an empty graph
with g.as_default(): #Set it as the default graph
    #Add two float scalar input nodes to the graph
    a = tf.placeholder(tf.float32, [], 'a') #float scalar
    b = tf.placeholder(tf.float32, [], 'b') #float scalar

    #Add an addition node to the graph
    c = a + b

    #Make the graph read-only
    g.finalize()

    tf.summary.FileWriter('.', g) #For TensorBoard visualisation

    #Create a session for the default graph
    with tf.Session() as s:
        [ result ] = s.run([ c ], { a: 1.0, b: 2.0 })
        print(result)
