import numpy as np

from libs.solver import Solver
from utils import helpers
from utils.transformer import get_transformer
from expt.common import synthetic_params


# Get data
transformer = get_transformer('synthesis')
df, _ = helpers.get_dataset('synthesis', params=synthetic_params)
y = df['label'].to_numpy()
X = df.drop('label', axis=1)
X = transformer.transform(X)

# Solver
s = Solver(X, y)
A, S, d = s.compute_matrix(X[0])
w, v = s.find_eig(S)

# Best response
z_p = s.best_response(w, v, d, k=50, period=99)
print(z_p)
print(z_p.shape)
print(z_p[0, :].shape)
z = np.zeros(len(d))
print(z.shape)
for i in range(99):
    z = np.logical_or(z, z_p[i, :])
print(sum(z))
print(np.where(z == 1)[0])

# Dual program
z_p = s.dp(w, v, d, k=50, step_size=0.1, period=95)
print(z_p)
z = np.zeros(len(d))
print(z.shape)
for i in range(95):
    z = np.logical_or(z, z_p[i, :])
print(sum(z))
print(np.where(z == 1)[0])
