import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt

###################################

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

            self.init = tf.global_variables_initializer()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def predict(self, x):
        return self.sess.run([ self.y ], { self.x: x })[0]

###################################

model = Model(2)
model.initialise()

xs = [ -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0 ]
ys = [ model.predict(x) for x in xs ] #Get y values of each of the x values.

(fig, ax) = plt.subplots(1, 1) #Set the figure with the given number of rows and columns of subplots. 'fig' control things about the whole figure whilst 'ax' controls things about the subplot within the figure.

ax.cla() #Optional: Clear the subplot.
ax.plot(xs, ys, color='red', linestyle='-', linewidth=3) #Plot the points with lines in between them (you can also use 'marker' and 'markersize' if you don't want lines).
ax.set_title('Polynomial') #Optional: Set the title for the subplot.
ax.set_xlim(-2.0, 2.0) #Optional: Set the range for the x-axis.
ax.set_xlabel('x') #Optional: Set the label for the x-axis.
ax.set_ylim(-10.0, 10.0) #Optional: Set the range for the y-axis.
ax.set_ylabel('y') #Optional: Set the label for the y-axis.
ax.grid(True) #Optional: Show a grid.

fig.tight_layout() #Optional: Resize the figure to reduce wasted space.
fig.show() #Show the figure.

model.close()

'''
Note: The subplots function returns different things for 'ax' based on the shape of subplots requested.

#Figure with two subplots side-by-side.
(fig, axs) = plt.subplots(1, 2)
ax1 = axs[0]
ax2 = axs[1]

#Figure with four subplots in a two-by-two grid.
(fig, axs) = plt.subplots(2, 2)
ax11 = axs[0,0]
ax12 = axs[0,1]
ax21 = axs[1,0]
ax22 = axs[1,1]
'''