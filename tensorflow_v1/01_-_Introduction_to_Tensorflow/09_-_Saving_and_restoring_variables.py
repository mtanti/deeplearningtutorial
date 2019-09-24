import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

g = tf.Graph()
with g.as_default():
    v = tf.get_variable('v', [], tf.float32)

    v_in = tf.placeholder(tf.float32, [], 'v_in')
    v_setter = tf.assign(v, v_in)
    
    #Create a saver node.
    saver = tf.train.Saver()

    g.finalize()

    with tf.Session() as s:
        s.run([ v_setter ], { v_in: 1.0 })
        
        #Save session in a bunch of files with names starting with 'model.'.
        saver.save(s, './model') #You can also add a folder here by using 'folder/model' instead of './model'.

    #Closing the previous session will lose the variable's value.
    with tf.Session() as s:
        #Restore variables.
        saver.restore(s, './model') #Be sure to also mention the folder here if you were saving inside a folder.

        [ result ] = s.run([ v ], { })
        print(result)