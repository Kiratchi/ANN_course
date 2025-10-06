import numpy as np

# Creating state space
x = np.array([
    [-1,  1, -1,  1, -1,  1, -1,  1],
    [-1, -1,  1,  1, -1, -1,  1,  1],
    [-1,  1,  1, -1,  1, -1, -1,  1]
])


M = 4
p_0 = 1
beta = 1


N = 3

# Initialize weights and thresholds
W = np.random.normal(loc=0, scale=1, size=(N, M))
theta_h = np.zeros((M,1)) 
theta_v = np.zeros((N,1)) 

# Repeat outer loop ??? times

# Sample 
cases = [0,1,2,3]
samples = np.random.choice(cases, size=p_0)


def p(b,beta):
    return 1 / ( 1+ np.exp(-2 * beta * b))   #Should "2" be here?

def evaluate_p(p):
    rand = np.random.rand(*p.shape)
    return np.where(rand < p, 1, -1)

k = 2

for mu, sample in enumerate(samples):
    v = x[:,sample].reshape(-1,1)
    b_h = W.T @ v - theta_h
    p_h = p(b_h, beta)
    h = evaluate_p(p_h)
     
    for t in range(k):
        print(W.shape, h.shape)
        b_v = h @ W - theta_v
        print(b_v)
    
