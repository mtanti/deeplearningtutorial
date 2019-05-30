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

#Diagram of neural network:

#     W1 W2 W3
# x_0 --O--O--O a3_0
#     \/ \/ \/
#     /\ /\ /\
# x_1 --O--O--O a3_1
#    l  k  j  i

#Forward pass

#e_i  = (a3_i - t_i)^2
#a3_i = sig(z3_i)
#z3_i = sum_j(a2_j*W3_ji) + b3_i
#a2_i = sig(z2_i)
#z2_i = sum_j(a1_j*W2_ji) + b2_i
#a1_j = sig(z1_j)
#z1_j = sum_k(x_k*W1_kj) + b1_j

e  = lambda i:(a3(i) - ts[:,i])**2
a3 = lambda i:sig(z3(i))
z3 = lambda i:sum(a2(j)*W3[j,i] for j in range(num_hiddens2)) + b3[i]
a2 = lambda j:sig(z2(j))
z2 = lambda j:sum(a1(k)*W2[k,j] for k in range(num_hiddens1)) + b2[j]
a1 = lambda k:sig(z1(k))
z1 = lambda k:sum(xs[:,l]*W1[l,k] for l in range(num_inputs)) + b1[k]

#Backward pass

#de_i/dW3_ji  = 2(a3_i - t_i) * da3_i/dW3_ji
#da3_i/dW3_ji = sig'(z3_i) * dz3_i/dW3_ji
#dz3_i/dW3_ji = a2_j

de_dW3  = lambda i,j:2*(a3(i) - ts[:,i]) * da3_dW3(i,j)
da3_dW3 = lambda i,j:sig_d(z3(i)) * dz3_dW3(j)
dz3_dW3 = lambda j  :a2(j)

#de_i/db3_i  = 2(a3_i - t_i) * da3_i/db3_i
#da3_i/db3_i = sig'(z3_i) * dz3_i/db3_i
#dz3_i/db3_i = 1

de_db3  = lambda i:2*(a3(i) - ts[:,i]) * da3_db3(i)
da3_db3 = lambda i:sig_d(z3(i)) * dz3_db3()
dz3_db3 = lambda  :1

#de_i/dW2_kj  = 2(a3_i - t_i) * da3_i/dW2_kj
#da3_i/dW2_kj = sig'(z3_i) * dz3_i/dW2_kj
#dz3_i/dW2_kj = W3_ji * da2_j/dW2_kj
#da2_j/dW2_kj = sig'(z2_j) * dz2_j/dW2_kj
#dz2_j/dW2_kj = a1_k

de_dW2  = lambda i,j,k:2*(a3(i) - ts[:,i]) * da3_dW2(i,j,k)
da3_dW2 = lambda i,j,k:sig_d(z3(i)) * dz3_dW2(i,j,k)
dz3_dW2 = lambda i,j,k:W3[j,i] * da2_dW2(j,k)
da2_dW2 = lambda j,k  :sig_d(z2(j)) * dz2_dW2(k)
dz2_dW2 = lambda k    :a1(k)

#de_i/db2_j  = 2(a3_i - t_i) * da3_i/db2_j
#da3_i/db2_j = sig'(z3_i) * dz3_i/db2_j
#dz3_i/db2_j = W3_ji * da2_j/db2_j
#da2_j/db2_j = sig'(z2_j) * dz2_j/db2_j
#dz2_j/db2_j = 1

de_db2  = lambda i,j:2*(a3(i) - ts[:,i]) * da3_db2(i,j)
da3_db2 = lambda i,j:sig_d(z3(i)) * dz3_db2(i,j)
dz3_db2 = lambda i,j:W3[j,i] * da2_db2(j)
da2_db2 = lambda j  :sig_d(z2(j)) * dz2_db2()
dz2_db2 = lambda    :1

#de_i/dW1_lk  = 2(a3_i - t_i) * da3_i/dW1_lk
#da3_i/dW1_lk = sig'(z3_i) * dz3_i/dW1_lk
#dz3_i/dW1_lk = sum_j[W3_ji * da2_j/dW1_lk]
#da2_j/dW1_lk = sig'(z2_j) * dz2_j/dW1_lk
#dz2_j/dW1_lk = W2_kj * da1_k/dW1_lk
#da1_k/dW1_lk = sig'(z1_k) * dz1_k/dW1_lk
#dz1_k/dW1_lk = x_l

de_dW1  = lambda i,k,l:2*(a3(i) - ts[:,i]) * da3_dW1(i,k,l)
da3_dW1 = lambda i,k,l:sig_d(z3(i)) * dz3_dW1(i,k,l)
dz3_dW1 = lambda i,k,l:sum(W3[j,i] * da2_dW1(j,k,l) for j in range(num_hiddens2))
da2_dW1 = lambda j,k,l:sig_d(z2(j)) * dz2_dW1(j,k,l)
dz2_dW1 = lambda j,k,l:W2[k,j] * da1_dW1(k,l)
da1_dW1 = lambda k,l  :sig_d(z1(k)) * dz1_dW1(l)
dz1_dW1 = lambda l    :xs[:,l]

#de_i/db1_k  = 2(a3_i - t_i) * da3_i/db1_k
#da3_i/db1_k = sig'(z3_i) * dz3_i/db1_k
#dz3_i/db1_k = sum_j[W3_ji * da2_j/db1_k]
#da2_j/db1_k = sig'(z2_j) * dz2_j/db1_k
#dz2_j/db1_k = W2_kj * da1_k/db1_k
#da1_k/db1_k = sig'(z1_k) * dz1_k/db1_k
#dz1_k/db1_k = 1

de_db1  = lambda i,k:2*(a3(i) - ts[:,i]) * da3_db1(i,k)
da3_db1 = lambda i,k:sig_d(z3(i)) * dz3_db1(i,k)
dz3_db1 = lambda i,k:sum(W3[j,i] * da2_db1(j,k) for j in range(num_hiddens2))
da2_db1 = lambda j,k:sig_d(z2(j)) * dz2_db1(j,k)
dz2_db1 = lambda j,k:W2[k,j] * da1_db1(k)
da1_db1 = lambda k  :sig_d(z1(k)) * dz1_db1()
dz1_db1 = lambda    :1

for epoch in range(num_epochs):
    #Assuming that the individual errors (for each input vector and each separate output) need to be added together to form the final error,
    # the individual error gradients also need to be added together.
    #d/dx (f(x) + g(x)) = d/dx f(x) + d/dx g(x)
    g_W3 = np.array([
                [ de_dW3(0,0).sum(), de_dW3(1,0).sum() ],
                [ de_dW3(0,1).sum(), de_dW3(1,1).sum() ]
            ])
    g_b3 = np.array([ de_db3(0).sum(), de_db3(1).sum() ])
    g_W2 = np.array([
                [ de_dW2(0,0,0).sum() + de_dW2(1,0,0).sum(), de_dW2(0,1,0).sum() + de_dW2(1,1,0).sum() ],
                [ de_dW2(0,0,1).sum() + de_dW2(1,0,1).sum(), de_dW2(0,1,1).sum() + de_dW2(1,1,1).sum() ]
            ])
    g_b2 = np.array([ de_db2(0,0).sum() + de_db2(1,0).sum(), de_db2(0,1).sum() + de_db2(1,1).sum() ])
    g_W1 = np.array([
                [ de_dW1(0,0,0).sum() + de_dW1(1,0,0).sum(), de_dW1(0,1,0).sum() + de_dW1(1,1,0).sum() ],
                [ de_dW1(0,0,1).sum() + de_dW1(1,0,1).sum(), de_dW1(0,1,1).sum() + de_dW1(1,1,1).sum() ]
            ])
    g_b1 = np.array([ de_db1(0,0).sum() + de_db1(1,0).sum(), de_db1(0,1).sum() + de_db1(1,1).sum() ])
    
    W3 -= learning_rate*g_W3
    b3 -= learning_rate*g_b3
    W2 -= learning_rate*g_W2
    b2 -= learning_rate*g_b2
    W1 -= learning_rate*g_W1
    b1 -= learning_rate*g_b1
    
    if epoch%10 == 0:
        print(epoch, e(0).sum() + e(1).sum())

ys = np.stack([ a3(0), a3(1) ], axis=1)

print()
for (x, y) in zip(xs.tolist(), ys.tolist()):
    print(x, np.round(y, 2))