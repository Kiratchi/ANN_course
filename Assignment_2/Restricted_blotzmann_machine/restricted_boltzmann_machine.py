# %%

import numpy as np
import pandas as pd
from plotnine import *
from math import log2, floor

# Creating state space
x = np.array([
    [-1,  1, -1,  1, -1,  1, -1,  1],
    [-1, -1,  1,  1, -1, -1,  1,  1],
    [-1, -1, -1, -1,  1,  1,  1,  1]
])
xor_indices = [0,3,5,6]
p_data = np.zeros(8, dtype=float)
p_data[xor_indices] = 0.25



def p(b, beta):
    z = 2.0 * beta * b
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def sample_from_p(p):
    rand = np.random.rand(*p.shape)
    return np.where(rand < p, 1, -1)

def sample_hidden(v, W, theta_h, beta):    
    b_h = W.T @ v - theta_h
    h = sample_from_p(p(b_h, beta)) 
    return h, b_h

def sample_visible(h, W, theta_v, beta):
    b_v = W @ h - theta_v
    v = sample_from_p(p(b_v,beta))
    return v, b_v

def run_one_chain(v0, W, theta_v, theta_h, beta, sampling_steps):
    v = v0.copy()
    h, b_h = sample_hidden(v, W, theta_h, beta)
    for t in range(sampling_steps):
        v, b_v = sample_visible(h, W, theta_v, beta)
        h, b_h = sample_hidden(v, W, theta_h, beta)
    return v, h, b_v, b_h


def run_boltzmann(x, xor_indices, 
                  M,
                  n_epoch = 4_000, 
                  sampling_steps = 10, 
                  batch_size = 16, 
                  beta = 2.5, 
                  eta = 0.02,
                  seed = None):

    N = x.shape[0]

    if seed is not None:
        np.random.seed(seed)

    # Initialize weights and thresholds
    W = np.random.normal(0, 0.05, size=(N, M))
    theta_h = np.zeros((M,1)) 
    theta_v = np.zeros((N,1)) 


    for nu in range(n_epoch):
        dW = np.zeros_like(W)
        dtheta_v = np.zeros_like(theta_v)
        dtheta_h = np.zeros_like(theta_h)

        samples = np.random.choice(xor_indices, size=batch_size, replace=True)
        for sample in samples:
            v0 = x[:,sample].reshape(-1,1)
            h0, b0_h = sample_hidden(v0, W, theta_h, beta)
            h0_mean = np.tanh(beta * b0_h)
            h0_mean = np.clip(h0_mean, -0.999, 0.999)

            vk, hk, bk_v, bk_h = run_one_chain(v0, W, theta_v, theta_h, beta, sampling_steps)
            hk_mean = np.tanh(beta * bk_h)
            hk_mean = np.clip(hk_mean, -0.999, 0.999)
            
            dW       += (v0 @ h0_mean.T) - (vk @ hk_mean.T)
            dtheta_v += -(v0 - vk)
            dtheta_h += -(h0_mean - hk_mean)


        W       += eta * dW / batch_size
        theta_v += eta * dtheta_v / batch_size
        theta_h += eta * dtheta_h / batch_size 
        
        # Weight decay
        W *= (1.0 - 1e-5) 

        # Print outs (heavy computation for p_model)
        # if nu % 500 == 0:
        #     p_model, _ = estimate_p_model(W, theta_v, theta_h, x, n_collect=5000, beta=beta)
        #     print(f"Epoch {nu}, sum XOR p = {p_model[xor_indices].sum():.3f}, mean|W| = {np.mean(np.abs(W))}")

    return W, theta_v, theta_h




def v_to_index(v):
    return int((v[0,0]+1)//2 + (v[1,0]+1) + 2*(v[2,0]+1))

def estimate_p_model(W, theta_v, theta_h, 
                    x,
                    beta = 2.5, 
                    n_burn = 10_000,
                    n_collect = 50_000,        # 10 000
                    seed = None):
    N = theta_v.shape[0]

    if seed is not None:
        np.random.seed(seed)
    
    rand_idx = np.random.randint(x.shape[1])
    v = x[:,rand_idx].reshape(-1,1)
    
    # Burn in
    for _ in range(n_burn):
        h, b_h = sample_hidden(v, W, theta_h, beta)
        v, b_v = sample_visible(h, W, theta_v, beta)
    
    # Sampling distribution
    x_indices = np.empty(n_collect, dtype=np.int64)
    for t in range(n_collect):
        h, b_h = sample_hidden(v, W, theta_h, beta)
        v, b_v = sample_visible(h, W, theta_v, beta) 
        x_indices[t] = v_to_index(v)

    counts = np.bincount(x_indices, minlength=2**N)
    p_model = counts / counts.sum()
    return p_model, x_indices

# CHATGPT function
def running_probs_long(x_indices, xor_indices, 
                       n_states=8, eps=1e-12):
    x = np.asarray(x_indices, int)
    T = x.size
    step = np.arange(1, T + 1)

    # one-hot encode and running probabilities
    onehot = (x[:, None] == np.arange(n_states)).astype(int)
    run = np.cumsum(onehot, axis=0) / step[:, None]

    # tidy DataFrame
    df = pd.DataFrame({
        'step':    np.repeat(step, n_states),
        'pattern': np.tile(np.arange(n_states), T),
        'p_hat':   run.ravel()
    })

    # mark XOR vs non-XOR
    df['In XOR'] = df['pattern'].isin(xor_indices)

    df['log_p_hat'] = np.log10(df['p_hat']+eps)
    return df


def plot_running_probs_all(x_indices, xor_indices, 
                           show_from=0, thin=1):
                           
    df_long = running_probs_long(x_indices, xor_indices, n_states=8)
    df_long = df_long[df_long['step'] >= show_from]
    if thin > 1:
        df_long = df_long[df_long['step'] % thin == 0]
    
    figure = (ggplot(df_long, aes('step', 'p_hat', color='factor(pattern)', linetype='In XOR'))
         + geom_line()
         + scale_linetype_manual(values={True: 'solid', False: 'dashed'})
         + labs(x='samples collected', 
                y='running probability', 
                color='pattern',
                title='Running probability for all 8 patterns')
         + theme_minimal()
         + theme(figure_size=(9, 4)))
    return figure

def D_kl(p_data, p_model, eps=1e-12):
    p_data_fix = np.maximum(p_data, eps)
    p_model_fix = np.maximum(p_model, eps)
    return np.sum(p_data * (np.log(p_data_fix)-np.log(p_model_fix)))


# %%

W, theta_v, theta_h = run_boltzmann(x, xor_indices, M=8, seed=42)
p_model, x_indices = estimate_p_model(W, theta_v, theta_h, x)
print(p_model)


# %%

figure = plot_running_probs_all(x_indices, xor_indices, show_from=0, thin=100)
out_png = "model-estimation.png"
figure.save(out_png, dpi=150)
figure



# %%
Ms = [1,2,4,8]
D_kl_list = []
rng_seeds = [12, 22, 32, 42]
for M,seed  in zip(Ms, rng_seeds):
    print(f"\n ### M={M} ###")
    W, theta_v, theta_h = run_boltzmann(x, xor_indices, M=M, seed=seed)
    p_model, x_indices = estimate_p_model(W, theta_v, theta_h, x)
    D_kl_list.append(D_kl(p_data, p_model))

df_D_kl = pd.DataFrame({'M': Ms, 'KL': D_kl_list})
print(df_D_kl)

# %%
print(df_D_kl)


def KL_theory(M, N=3):
    # Equation: N - int(log2(M+1)) - (M+1)/2^(int(log2(M+1))) for M < 2^(N-1)-1, else 0
    M = np.asarray(M)
    result = []
    for m in M:
        if m < 2**(N-1) - 1:
            k = int(floor(log2(m+1)))
            val = N - k - (m+1) / (2**k)
        else:
            val = 0
        result.append(val)
    return np.array(result, dtype=float)

# Generate theory values for a smooth red curve
M_theory = np.arange(1, 10)
KL_theory_vals = KL_theory(M_theory)
df_theory = pd.DataFrame({'M': M_theory, 'KL_theory': KL_theory_vals})


figure = (
    ggplot()
    + geom_line(df_theory, aes('M', 'KL_theory', linetype='"Theoretical KL Divergence"'),
            color='blue', size=1)
    + geom_line(df_D_kl, aes('M', 'KL', color='"KL Divergence"'), size=1)
    + geom_point(df_D_kl, aes('M', 'KL', color='"KL Divergence"'), size=5)
    + labs(
        title='Kullback-Leibler Divergence vs Number of Hidden Neurons',
        x='Number of Hidden Neurons (M)',
        y='KL Divergence',
        color='',
        linetype=''
    )
    + theme_minimal()
    + theme(
        legend_position='top',
        figure_size=(7,5),
        text=element_text(size=12)
    )
)
figure
figure.save("D_KL.png", dpi=300, width=6, height=4)

# %%
