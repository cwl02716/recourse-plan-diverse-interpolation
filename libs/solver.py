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

    def compute_matrix(self, x_0):
        A = np.zeros((self.dim, self.N))
        d = np.zeros(self.N)

        for i in range(self.N):
            A[:, i] = (self.data_hat[i] - x_0) / (np.linalg.norm(self.data_hat[i] - x_0))
            d[i] = np.exp(-np.linalg.norm(self.data_hat[i] - x_0) ** 2 / self.h ** 2)

        S = np.dot(A.T, A)

        return A, S, d

    def find_eig(self, matrix):
        w, v = eigh(matrix)
        sum_eig = sum(w ** 2)
        cur_sum = sum_eig

        for i in range(len(w) - 1, -1, -1):
            cur_sum -= w[i] ** 2
            if cur_sum / sum_eig < 0.1:
                print(cur_sum / sum_eig)
                return w[i:len(w)], v[i:len(w), :]

    def best_response(self, w, v, d, k, max_iter=100, period=80):
        z = np.zeros(self.N)
        z_p = np.zeros((period, self.N))

        m_dim = len(w)
        gamma = np.zeros(m_dim)

        for i in range(max_iter):
            for j in range(m_dim):
                gamma[j] = -(self.theta ** 2 * w[j] * np.dot(v[j], z))

            gamma_identity = (1 - self.theta) * d - 2 * self.theta * np.dot(v.T, gamma)
            idx = (-gamma_identity).argsort()[:k]
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
                gamma_add[j] = kappa * ((-2 * gamma[j]) / (self.theta * w[j]) - 2 * self.theta * np.dot(v[j], z))

            gamma += gamma_add
            gamma_identity = (1 - self.theta) * d - 2 * self.theta * np.dot(v.T, gamma)
            idx = (-gamma_identity).argsort()[:k]
            z = np.zeros(self.N)
            z[idx] = 1

            if i > max_iter - period:
                z_p[i - max_iter + period, :] = z

        return z_p
