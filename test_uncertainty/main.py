import numpy as np

def getDagger(S_flat, threshold=1e-10):
    
    S_dagger_flat = np.zeros_like(S_flat)
    valid_idx = S_flat > threshold
    S_dagger_flat[valid_idx] = S_dagger_flat[valid_idx]**(-1)

    return S_dagger_flat
   

ens_member = 31
n_pts = 5
x = np.linspace(0, 1, n_pts) # spatial coordinate

resamples = 100

sigma_f = 1.0  # This is the SPREAD of model variable, NO uncertainty but with spread
sigma_y = 1.0  # This is the uncertainty of y

f = np.zeros((len(x), ens_member))
for i in range(ens_member):
    f[:, i] = x ** 2.0 + np.random.randn(len(x)) * sigma_f
    
f -= np.mean(f, axis=1, keepdims=True)
G = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
])

y = G @ f

true_data = dict(
    f = f,
    G = G,
    y = y,
    svd = np.linalg.svd(f),
)


G_ens = np.zeros((resamples, *G.shape))

for i in range(resamples):

    
    print("Doing sampling %d" % (i+1,))
    
    y_obs = y + np.random.randn(*y.shape) * sigma_y

    svd_obs = np.linalg.svd(f)
    S_obs_flat = svd_obs.S
    U_obs  = svd_obs.U    
    Vh_obs = svd_obs.Vh
    
    S_dagger_obs_flat = getDagger(S_obs_flat)
    S_dagger_obs = np.zeros((U_obs.shape[0], Vh_obs.shape[0]))
    for j, val in enumerate(S_dagger_obs_flat):
        S_dagger_obs[j, j] = val
    
    df = f_obs   - true_data["f"]
    dy = y_obs   - true_data["y"]
    dU = U_obs   - true_data["svd"].U
    dVh = Vh_obs - true_data["svd"].Vh
    
    G_obs = np.transpose(U_obs @ S_dagger_obs @ Vh_obs @ y_obs.T)
    
    G_ens[i, :, :] = G_obs


# uncertainty





#return f + np.random.randn(f.shape) * sigma 
    

# y = Gf





