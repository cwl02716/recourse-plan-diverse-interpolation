import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from libs.solver import Solver
from utils import helpers
from utils.transformer import get_transformer
from expt.common import synthetic_params
from classifiers.mlp import Net0


# Get data
transformer = get_transformer('synthesis')
df, _ = helpers.get_dataset('synthesis', params=synthetic_params)
y = df['label'].to_numpy()
X = df.drop('label', axis=1)
X = transformer.transform(X)

# Get model
with open("results/run_0/checkpoints/mlp_synthesis_5.pickle", "rb") as f:
    model = pickle.load(f)[0]
y = model.predict(X)

# Solver
best_response = True
# theta_l = [0.1 * i for i in range(11)]
theta_l = [1]
# kernel_l = [0.1 * (i + 1) for i in range(21, 40)]
kernel_l = [0.5]
for idx in range(len(theta_l)):
    for j in range(len(kernel_l)):
        s = Solver(model, X, y, theta=1, kernel_width=kernel_l[j])
        recourse_set, X_diverse, X_other = s.generate_recourse(np.array([1.5, 2]), k=3)
        # A, S, d = s.compute_matrix(np.array([1.5, 2]), X[y == 1])
        # w, v = s.find_eig(S)

        # Best response or dp
        # if best_response:
        #     z_p = s.best_response(w, v, d, k=5, period=20)
        # else:
        #     z_p = s.dp(w, v, d, k=5, step_size=1, period=20)
    
        # z_prev = np.zeros(len(d))

        # for i in range(20):
        #     z_prev = np.logical_or(z_prev, z_p[i, :])
    
        # idx_l = np.where(z_prev == 1)[0]

        # data = X[y == 1][np.where(z_prev == 1)[0]]
        # A, S, d = s.compute_matrix(X[0], data)
        # z = s.quad(S, d, 3)
        # idx = idx_l[np.where(z == 1)[0]]

        # X_diverse = X[y == 1][idx]
        # X_other = X[y == 1][np.where(z_prev == 0)[0]]

        # Plot
        fig, ax = plt.subplots()
        # ax.scatter(X[0][0], X[0][1], s=100, marker='*', label = 'x0')
        ax.scatter(1.5, 2, s=100, marker='*', label = 'x0')
        ax.scatter(recourse_set[:, 0], recourse_set[:, 1], marker='o', c='#17becf', label = 'Recourse')
        ax.scatter(X_diverse[:, 0], X_diverse[:, 1], marker='o', label = 'Diverse')
        ax.scatter(X_other[:, 0], X_other[:, 1], marker='o', label = 'Other')

        ax.set(xlabel='$x_{0}$', ylabel='$x_{1}$')
        ax.legend(loc='upper right', frameon=False)

        ax.set_xlim(-4, 8)
        ax.set_ylim(-4, 8)
        plt.tight_layout()

        if not os.path.exists('results/figures/'):
            os.makedirs('results/figures/')
        plt.savefig(f'results/figures/visualization_BR.jpg', dpi=500, transparent=True)
        plt.show()
