import tensorflow as tf

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, [2, 3], 'x') #float matrix of 2 rows and 3 columns

    #Get the shape of a tensor
    shape = tf.shape(x)
    
    #Get a sub-tensor
    subtensor = x[1, 1:]
    
    #Concatenate two tensors together
    concat = tf.concat([ x, x ], axis=1)
    
    #Reshape a tensor
    reshape = tf.reshape(x, [6, 1])
    
    #Tile a tensor
    tile = tf.tile(x, [2, 2])
    
    g.finalize()

    with tf.Session() as s:
        [ result_shape, result_subtensor, result_concat, result_reshape, result_tile ] = s.run([ shape, subtensor, concat, reshape, tile ], { x: [ [ 1.0, 2.0, 3.0 ], [ 4.0, 5.0, 6.0 ] ] })
        
        print('shape')
        print(result_shape)
        print()
        
        print('sub-tensor')
        print(result_subtensor)
        print()
        
        print('concatenate')
        print(result_concat)
        print()
        
        print('reshape')
        print(result_reshape)
        print()
        
        print('tile')
        print(result_tile)