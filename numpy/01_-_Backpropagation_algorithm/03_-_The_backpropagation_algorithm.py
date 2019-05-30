import numpy as np

def sig(z):
    return 1/(1 + np.exp(-z))

def sig_d(z):
    return np.exp(z)/(1 + np.exp(z))**2

num_inputs   = 2
num_hiddens1 = 2
num_hiddens2 = 2
num_outputs  = 2

learning_rate = 2.5
num_epochs = 200

xs = np.array([ [0,0], [0,1], [1,0], [1,1] ])
ts = np.array([ [0,0], [0,1], [1,0], [1,1] ])

W1 = np.random.normal(0.0, 1.0, size=[num_inputs, num_hiddens1])
b1 = np.zeros([num_hiddens1])

W2 = np.random.normal(0.0, 1.0, size=[num_hiddens1, num_hiddens2])
b2 = np.zeros([num_hiddens2])

W3 = np.random.normal(0.0, 1.0, size=[num_hiddens2, num_outputs])
b3 = np.zeros([num_outputs])

# Diagram of neural network:

#     W1 W2 W3
# x_0 --O--O--O a3_0
#     \/ \/ \/
#     /\ /\ /\
# x_1 --O--O--O a3_1

#Extracting a pattern:

#de_i/dW3_ji = 2(a3_i - t_i)*sig'(z3_i) * a2_j
#de_i/dW2_kj = 2(a3_i - t_i)*sig'(z3_i) * W3_ji*sig'(z2_j) * a1_k
#de_i/dW1_lk = 2(a3_i - t_i)*sig'(z3_i) * W3_ji*sig'(z2_j) * W2_kj*sig'(z1_k) * x_l
#
#de_i/db3_i  = 2(a3_i - t_i)*sig'(z3_i)
#de_i/db2_j  = 2(a3_i - t_i)*sig'(z3_i) * W3_ji*sig'(z2_j)
#de_i/db1_k  = 2(a3_i - t_i)*sig'(z3_i) * W3_ji*sig'(z2_j) * W2_kj*sig'(z1_k)

#We can define a recursive function that computes the gradient for a whole layer based on the parts of the gradient for the next layer (starting from the output layer).
#For a single input vector-
#Output layer (base case):
#  d3 = de/da3 * da3/dz3
#  de/dW3 = a2.T matmul d3
#  de/db3 = d3
#All other layers (recursive case):
#  dp = (d{p+1} matmul W{p+1}.T) * dap/dzp
#  de/dWp = a{p-1}.T matmul d{p}
#  de/dbp = d{p}

for epoch in range(num_epochs):
    g_W3 = np.zeros_like(W3)
    g_b3 = np.zeros_like(b3)
    g_W2 = np.zeros_like(W2)
    g_b2 = np.zeros_like(b2)
    g_W1 = np.zeros_like(W1)
    g_b1 = np.zeros_like(b1)
    e = 0
    for (x, t) in zip(xs, ts):
        #forward pass
        
        z1 = np.dot(x, W1) + b1
        a1 = sig(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sig(z2)
        z3 = np.dot(a2, W3) + b3
        a3 = sig(z3)

        e += ((a3 - t)**2).sum()

        #backward pass

        d3 = 2*(a3 - t) * sig_d(z3)
        g_W3 += np.dot(a2.reshape([1,-1]).T, d3.reshape([1,-1]))
        g_b3 += d3

        d2 = np.dot(d3, W3.T) * sig_d(z2)
        g_W2 += np.dot(a1.reshape([1,-1]).T, d2.reshape([1,-1]))
        g_b2 += d2

        d1 = np.dot(d2, W2.T) * sig_d(z1)
        g_W1 += np.dot(x.reshape([1,-1]).T, d1.reshape([1,-1]))
        g_b1 += d1

    W3 -= learning_rate*g_W3
    b3 -= learning_rate*g_b3
    W2 -= learning_rate*g_W2
    b2 -= learning_rate*g_b2
    W1 -= learning_rate*g_W1
    b1 -= learning_rate*g_b1
    
    if epoch%10 == 0:
        print(epoch, e)

z1 = np.dot(xs, W1) + b1
a1 = sig(z1)
z2 = np.dot(a1, W2) + b2
a2 = sig(z2)
z3 = np.dot(a2, W3) + b3
a3 = sig(z3)

ys = a3

print()
for (x, y) in zip(xs.tolist(), ys.tolist()):
    print(x, np.round(y, 2))