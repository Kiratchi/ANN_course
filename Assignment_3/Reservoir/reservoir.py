import pandas as pd
import numpy as np

######## WEIGHT INITIATION ###########

def initiate_input_weights(N_input = 3, N_reservoir = 500, var = 0.002, seed = None):
    if seed is not None:
        np.random.seed(seed)
    return(np.random.normal(loc=0.0, scale= np.sqrt(var), size=(N_reservoir,N_input)))

def initiate_reservoir_weights(N_reservoir = 500, var = 2/500, seed = None):
    if seed is not None:
        np.random.seed(seed)
    return(np.random.normal(loc=0.0, scale= np.sqrt(var), size=(N_reservoir,N_reservoir)))

######## RESERVOIR FUNCTIONS ###########

# Function to run reservoir
def forward_run(R, x, W_reservoir, W_input):
    # print(W_input.shape)
    # print(x.shape)
    b = W_reservoir @ R + W_input @ x
    return(np.tanh(b))

def set_output_weights(R, Y, k = 0.01):
    return(Y @ R.T @ np.linalg.inv(R @ R.T + k * np.eye(R.shape[0])))


######## LOAD AND PREPARE DATA ###########

X = pd.read_csv("training-set.csv", header=None).values
X_test  = pd.read_csv("test-set-10.csv", header=None).values

N_input = X.shape[0] 
T = X.shape[1]
N_reservoir = 500


######## TRAINING PHASE ###########

# Initialize random weights
W_input = initiate_input_weights()
W_reservoir = initiate_reservoir_weights()

# Run reservoir on training data
R_train = np.zeros((N_reservoir, T+1))
for t in range(T):
    x_t = X[:,t]
    R_train[:,t+1] = forward_run(R_train[:,t], x_t, W_reservoir, W_input)

# Set output weights using ridge regression
W_out = set_output_weights(R_train[:, 1:-1], X[:, 1:])


######## TEST PHASE ###########

R_test = np.zeros((N_reservoir, X_test.shape[1]+1))
for t in range(X_test.shape[1]):
    x_t = X_test[:, t]
    R_test[:, t+1] = forward_run(R_test[:, t], x_t, W_reservoir, W_input)


######## PREDICTION PHASE ###########

O_pred = np.zeros((3, 500))
R = R_test[:, -1]            # Last reservoir state 
x_t = X_test[:,-1]           # Last known input

for t in range(500):
    R = forward_run(R, x_t, W_reservoir, W_input)
    O_t = W_out @ R
    O_pred[:, t] = O_t
    x_t = O_t           #Output is new input

######## SAVE PREDICTIONS ###########

pd.DataFrame([O_pred[1, :]]).to_csv("prediction.csv", index=False, header=False)


