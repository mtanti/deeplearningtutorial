import numpy as np

# Diagram of neural network:
#
#     W1 W2 W3
# x_0 --O--O--O a3_0
#     \/ \/ \/
#     /\ /\ /\
# x_1 --O--O--O a3_1
#    l  k  j  i

def sig(z):
    return 1/(1 + np.exp(-z))

def sig_d(z):
    return np.exp(z)/(1 + np.exp(z))**2

num_inputs   = 2
num_hiddens1 = 2
num_hiddens2 = 2
num_outputs  = 2

epsilon = 1e-1
learning_rate = 2.0
num_epochs = 200

W1 = np.random.normal(0.0, 1.0, size=[num_inputs,num_hiddens1])
b1 = np.zeros([num_hiddens1])

W2 = np.random.normal(0.0, 1.0, size=[num_hiddens1,num_hiddens2])
b2 = np.zeros([num_hiddens2])

W3 = np.random.normal(0.0, 1.0, size=[num_hiddens2,num_outputs])
b3 = np.zeros([num_outputs])

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = np.array([[0, 0], [0, 1], [0, 1], [1, 0]])

#e_i  = (a3_i - t_i)^2
#a3_i = sig(z3_i)
#z3_i = sum_j(a2_j*W3_ji) + b3_i
#a2_i = sig(z2_i)
#z2_i = sum_j(a1_j*W2_ji) + b2_i
#a1_j = sig(z1_j)
#z1_j = sum_k(x_k*W1_kj) + b1_j

e  = lambda i:(a3(i) - t[:,i])**2
a3 = lambda i:sig(z3(i))
z3 = lambda i:sum(a2(j)*W3[j,i] for j in range(num_hiddens2)) + b3[i]
a2 = lambda j:sig(z2(j))
z2 = lambda j:sum(a1(k)*W2[k,j] for k in range(num_hiddens1)) + b2[j]
a1 = lambda k:sig(z1(k))
z1 = lambda k:sum(x[:,l]*W1[l,k] for l in range(num_inputs)) + b1[k]

for _ in range(num_epochs):
    print(e(0).sum() + e(1).sum())

    g_W3 = np.zeros_like(W3)
    for i in range(num_outputs):
        for j in range(num_hiddens2):
            W3_ij_orig = W3[j,i]
            W3[j,i] += epsilon
            e_plus = e(0).sum() + e(1).sum()
            W3[j,i] = W3_ij_orig
            W3[j,i] -= epsilon
            e_minus = e(0).sum() + e(1).sum()
            W3[j,i] = W3_ij_orig
            g_W3[j,i] = (e_plus - e_minus)/(2*epsilon)

    g_b3 = np.zeros_like(b3)
    for i in range(num_outputs):
        b3_i_orig = b3[i]
        b3[i] += epsilon
        e_plus = e(0).sum() + e(1).sum()
        b3[i] = b3_i_orig
        b3[i] -= epsilon
        e_minus = e(0).sum() + e(1).sum()
        b3[i] = b3_i_orig
        g_b3[i] = (e_plus - e_minus)/(2*epsilon)

    g_W2 = np.zeros_like(W2)
    for j in range(num_hiddens2):
        for k in range(num_hiddens1):
            W2_jk_orig = W2[k,j]
            W2[k,j] += epsilon
            e_plus = e(0).sum() + e(1).sum()
            W2[k,j] = W2_jk_orig
            W2[k,j] -= epsilon
            e_minus = e(0).sum() + e(1).sum()
            W2[k,j] = W2_jk_orig
            g_W2[k,j] = (e_plus - e_minus)/(2*epsilon)

    g_b2 = np.zeros_like(b2)
    for j in range(num_hiddens2):
        b2_j_orig = b2[j]
        b2[j] += epsilon
        e_plus = e(0).sum() + e(1).sum()
        b2[j] = b2_j_orig
        b2[j] -= epsilon
        e_minus = e(0).sum() + e(1).sum()
        b2[j] = b2_j_orig
        g_b2[j] = (e_plus - e_minus)/(2*epsilon)

    g_W1 = np.zeros_like(W1)
    for k in range(num_hiddens1):
        for l in range(num_inputs):
            W1_kl_orig = W1[l,k]
            W1[l,k] += epsilon
            e_plus = e(0).sum() + e(1).sum()
            W1[l,k] = W1_kl_orig
            W1[l,k] -= epsilon
            e_minus = e(0).sum() + e(1).sum()
            W1[l,k] = W1_kl_orig
            g_W1[l,k] = (e_plus - e_minus)/(2*epsilon)

    g_b1 = np.zeros_like(b1)
    for k in range(num_hiddens1):
        b1_k_orig = b1[k]
        b1[k] += epsilon
        e_plus = e(0).sum() + e(1).sum()
        b1[k] = b1_k_orig
        b1[k] -= epsilon
        e_minus = e(0).sum() + e(1).sum()
        b1[k] = b1_k_orig
        g_b1[k] = (e_plus - e_minus)/(2*epsilon)


    W3 -= learning_rate*g_W3
    b3 -= learning_rate*g_b3
    W2 -= learning_rate*g_W2
    b2 -= learning_rate*g_b2
    W1 -= learning_rate*g_W1
    b1 -= learning_rate*g_b1

print(np.stack([a3(0), a3(1)], 1))