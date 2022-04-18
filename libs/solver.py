import numpy as np
from scipy.linalg import eigh

import gurobipy as grb


class Solver(object):
    """ Class for optimization problem """

    def __init__(self, data, labels, theta=0.5, kernel_width=1):
        self.data = data
        self.labels = labels
        self.theta = theta
        self.h = kernel_width

        self.data_hat = self.data[self.labels == 1]
        self.N, self.dim = self.data_hat.shape

    def quad(self, S, d, k):
        """ Solve quadratic program with gurobi
        """
        dim = len(d)

        # Model initialization
        model = grb.Model("qcp")
        model.params.NonConvex = 2
        model.setParam('OutputFlag', False)
        model.params.threads = 64

        # Variables
        z = model.addMVar(dim, vtype=grb.GRB.BINARY, name="z")
        z_sub = model.addMVar(1, vtype=grb.GRB.CONTINUOUS, name="zsub")

        # Set objectives
        obj = (1 - self.theta) * d @ z + self.theta * z_sub
        model.setObjective(obj, grb.GRB.MINIMIZE)

         # Constraints
        model.addConstr(z.sum() == k)
        model.addConstr(z_sub == z @ S @ z)

        # Optimize
        model.optimize()

        z_opt = np.zeros(dim)

        for i in range(dim):
            z_opt[i] = z[i].x
        
        return z_opt

    def compute_matrix(self, x_0, data):
        N, dim = data.shape

        A = np.zeros((dim, N))
        d = np.zeros(N)

        for i in range(N):
            A[:, i] = (data[i] - x_0) / (np.linalg.norm(data[i] - x_0))
            d[i] = np.linalg.norm(data[i] - x_0)

        S = np.dot(A.T, A)

        return A, S, d

    def find_eig(self, matrix):
        w, v = eigh(matrix)
        sum_eig = sum(w ** 2)
        cur_sum = sum_eig

        for i in range(len(w) - 1, -1, -1):
            cur_sum -= w[i] ** 2
            if cur_sum / sum_eig < 1e-9:
                return np.flip(w[i:len(w)]), np.flip(v[:, i:len(w)], axis=1)

    def best_response(self, w, v, d, k, max_iter=100, period=80):
        z = np.zeros(self.N)
        z_p = np.zeros((period, self.N))

        m_dim = len(w)
        gamma = np.zeros(m_dim)

        for i in range(max_iter):
            for j in range(m_dim):
                gamma[j] = -(self.theta * w[j] * np.dot(v[:, j], z))

            gamma_identity = (1 - self.theta) * d - 2 * np.dot(v, gamma)
            idx = (gamma_identity).argsort()[:k]
            z = np.zeros(self.N)
            z[idx] = 1

            if i > max_iter - period:
                z_p[i - max_iter + period, :] = z

        return z_p

    def dp(self, w, v, d, k, step_size=0.1, max_iter=100, period=80):
        z = np.zeros(self.N)
        z_p = np.zeros((period, self.N))

        m_dim = len(w)
        gamma = np.zeros(m_dim)

        for i in range(max_iter):
            kappa = step_size / np.sqrt(i + 1)

            gamma_add = np.zeros(m_dim)
            for j in range(m_dim):
                gamma_add[j] = kappa * ((-2 * gamma[j]) / (self.theta * w[j]) - 2 * np.dot(v[:, j], z))

            gamma += gamma_add
            gamma_identity = (1 - self.theta) * d - 2 * np.dot(v, gamma)
            idx = (gamma_identity).argsort()[:k]
            z = np.zeros(self.N)
            z[idx] = 1

            if i > max_iter - period:
                z_p[i - max_iter + period, :] = z

        return z_p
