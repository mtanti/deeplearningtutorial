import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, [2, 3], 'x') #Float matrix of 2 rows and 3 columns named 'x'.

    #Get the shape of a tensor.
    shape = tf.shape(x)
    
    #Get a sub-tensor.
    subtensor = x[1, 1:]
    
    #Concatenate two tensors together.
    concat = tf.concat([ x, x ], axis=1)
    
    #Reshape a tensor.
    reshape = tf.reshape(x, [6, 1])
    
    #Tile a tensor.
    tile = tf.tile(x, [2, 2])
    
    g.finalize()

    with tf.Session() as s:
        X = [ [ 1.0, 2.0, 3.0 ], [ 4.0, 5.0, 6.0 ] ]
        
        print('shape')
        [ result ] = s.run([ shape ], { x: X })
        print(result)
        print()
        
        print('sub-tensor')
        [ result ] = s.run([ subtensor ], { x: X })
        print(result)
        print()
        
        print('concatenate')
        [ result ] = s.run([ concat ], { x: X })
        print(result)
        print()
        
        print('reshape')
        [ result ] = s.run([ reshape ], { x: X })
        print(result)
        print()
        
        print('tile')
        [ result ] = s.run([ tile ], { x: X })
        print(result)