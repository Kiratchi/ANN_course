
import pandas as pd
import numpy as np

# Initialize weights
def initiate_W(dim = (40,40,4)):
    return np.random.uniform(0,1,dim)

# idx for w with smallest Eulidean distance
def find_bmu(x_i, W):
    dist_squared = np.sum((W - x_i)**2, axis=2)
    return np.unravel_index(np.argmin(dist_squared), dist_squared.shape)

def h(r_i, r_i0, sigma):
    d = ( r_i[0]- r_i0[0])**2 + ( r_i[1]- r_i0[1])**2
    return np.exp( -d/( 2*sigma**2 ) )

def train_map(X, W_0, n_epoch=10, 
              eta_0=0.1, decay_eta=0.01, 
              sigma_0=10, decay_sigma=0.05): 
    W = W_0.copy()
    order = np.arange(0, X.shape[0])
    for epoch in range(n_epoch):
        eta = eta_0 * np.exp(-decay_eta * epoch)
        sigma = sigma_0 * np.exp(-decay_sigma * epoch)
        order = np.random.permutation(order)
        for i in order:
            x_i = X[i]
            bmu_idx = find_bmu(x_i, W)
            for j in range(0, W.shape[0]):
                for k in range(0, W.shape[1]):
                    W[j,k] += eta * h((j,k), bmu_idx, sigma) * (x_i - W[j,k]) 
    return W


####### IMPORTING DATA #######
X = pd.read_csv("iris-data.csv", header=None).values.astype(float)
y = pd.read_csv("iris-labels.csv", header=None).values.squeeze()


####### MAIN CODE #######
# Normalize X
X = X / np.max(X)
W_0 = initiate_W()
W = train_map(X, W_0)

bmus_before = np.array([find_bmu(x_i,W_0) for x_i in X])
bmus_after = np.array([find_bmu(x_i,W) for x_i in X])

print("Unique BMUs before:", len(np.unique(bmus_before, axis=0)))
print("Unique BMUs after :", len(np.unique(bmus_after,  axis=0)))


####### PLOTTING RESULTS #######
from plotnine import *

# Reformating data to data frame
df_before = pd.DataFrame({
    "row": bmus_before[:, 0],
    "col": bmus_before[:, 1],
    "label": y,
    "stage": "2 Initial weights"
})
df_after = pd.DataFrame({
    "row": bmus_after[:, 0],
    "col": bmus_after[:, 1],
    "label": y,
    "stage": "1 Final weights"
})
df_plot = pd.concat([df_after, df_before], ignore_index=True)
df_plot["label"] = df_plot["label"].astype("category")

# Adding species names
species_map = {0.0: "Iris setosa", 1.0: "Iris versicolor", 2.0: "Iris virginica"}
df_plot["species"] = df_plot["label"].map(species_map).astype("category")


p = (
    ggplot(df_plot, aes(x="col", y="row", color="species"))
    + geom_point(size=1.8, alpha=0.85, position=position_jitter(width=0.25, height=0.25))
    + scale_y_reverse()
    + coord_equal()
    + facet_grid("~stage")
    + labs(x="Column index", y="Row index", color="Species")
    + theme(figure_size=(10, 5))
)

p.save("SOM.png", dpi=120)