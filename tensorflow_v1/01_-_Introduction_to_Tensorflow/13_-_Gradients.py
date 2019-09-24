import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt

class Model(object):

    def __init__(self, degree):
        num_coefficients = degree + 1
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [], 'x')

            self.coeffs = []
            for i in range(num_coefficients):
                coeff = tf.get_variable('coeff_'+str(i), [], tf.float32, tf.random_normal_initializer())
                self.coeffs.append(coeff)
                if i == 0:
                    self.y = coeff
                else:
                    self.y = self.y + coeff*self.x**i

            #Add nodes to the graph that calculate the derivative of y with respect to x.
            [ self.grad ] = tf.gradients([ self.y ], [ self.x ])
            
            self.init = tf.global_variables_initializer()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def predict(self, x):
        return self.sess.run([ self.y ], { self.x: x })[0]
    
    def gradient(self, x):
        return self.sess.run([ self.grad ], { self.x: x })[0] #Get the gradient of y at the given x.

###################################

model = Model(3)
model.initialise()

xs = [ -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0 ]
ys = [ model.predict(x) for x in xs ]
gs = [ model.gradient(x) for x in xs ] #Get the gradient at every x.

(fig, axs) = plt.subplots(1, 2) #Show two subplots side-by-side.

#Show the polynomial on the left.
axs[0].plot(xs, ys, color='red', linestyle='-', linewidth=3)
axs[0].set_title('Polynomial')
axs[0].set_xlim(-2.0, 2.0)
axs[0].set_xlabel('x')
axs[0].set_ylim(-10.0, 10.0)
axs[0].set_ylabel('y')
axs[0].grid(True)

#Show the polynomial's gradient on the right.
axs[1].plot(xs, gs, color='blue', linestyle=':', linewidth=3)
axs[1].set_title('Gradient')
axs[1].set_xlim(-2.0, 2.0)
axs[1].set_xlabel('x')
axs[1].set_ylim(-10.0, 10.0)
axs[1].set_ylabel('dy/dx')
axs[1].grid(True)

fig.tight_layout()
fig.show()

model.close()