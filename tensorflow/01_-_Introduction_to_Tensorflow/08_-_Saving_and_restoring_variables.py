import tensorflow as tf

g = tf.Graph()
with g.as_default():
    v = tf.get_variable('v', [], tf.float32)

    v_in = tf.placeholder(tf.float32, [], 'v_in')
    v_setter = tf.assign(v, v_in) #Run this to set v to v_in
    
    #Create a saver node
    saver = tf.train.Saver()

    g.finalize()

    with tf.Session() as s:
        s.run([ v_setter ], { v_in: 1.0 })
        
        saver.save(s, './model') #Save session in a bunch of files with names starting with ‘model.’

    #Closing the previous session will lose the variable’s value
    with tf.Session() as s:
        saver.restore(s, './model') #Restore variables

        [ result ] = s.run([ v ], { })
        print(result)
