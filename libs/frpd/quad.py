import time
import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors

import gurobipy as grb


class Solver(object):
    """ Class for optimization problem """

    def __init__(self, model, data, labels, theta=0.5, kernel_width=1):
        self.model = model
        self.data = data
        self.labels = labels
        self.theta = theta
        self.h = kernel_width

        self.data_hat = self.data[self.labels == 1]
        self.N, self.dim = self.data_hat.shape

    def quad(self, S, d, k, cost_diverse=True):
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
        if cost_diverse:
            obj = (1 - self.theta) * d @ z + self.theta * z_sub
        else:
            obj = z_sub
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Constraints
        if cost_diverse:
            model.addConstr(z.sum() == k)
        else:
            for i in range(0, k * k, k):
                model.addConstr(z[i:i+k].sum() == 1)
        model.addConstr(z_sub == z @ S @ z)

        # Optimize
        model.optimize()

        z_opt = np.zeros(dim)

        for i in range(dim):
            z_opt[i] = z[i].x

        return z_opt

    def compute_matrix(self, x0, data):
        N, dim = data.shape

        A = np.zeros((dim, N))
        d = np.zeros(N)

        A = (data - x0).T / np.linalg.norm(data - x0, axis=1)
        d = np.linalg.norm(data - x0, axis=1)
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
            gamma = -self.theta * (np.multiply(w, np.dot(v.T, z)))
            # for j in range(m_dim):
                # gamma[j] = -(self.theta * w[j] * np.dot(v[:, j], z))

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
    
    def solve(self, x0, k, period=20, best_response=True, return_time=False):
        A, S, d = self.compute_matrix(x0, self.data[self.labels == 1])
        w, v = self.find_eig(S)

        # Best response and dp
        start_time = time.time()
        if best_response:
            z_p = self.best_response(w, v, d, k=k, period=period)
        else:
            z_p = self.dp(w, v, d, k=k, step_size=1, period=period)
        
        z_prev = np.zeros(len(d))
        for i in range(period):
            z_prev = np.logical_or(z_prev, z_p[i, :])
        
        t = time.time() - start_time

        idx_l = np.where(z_prev == 1)[0]

        data = self.data[self.labels == 1][np.where(z_prev == 1)[0]]
        A, S, d = self.compute_matrix(x0, data)
        z = self.quad(S, d, k)

        idx = idx_l[(-z).argsort()[:k]]
        X_diverse = self.data[self.labels == 1][idx]
        X_other = self.data[self.labels == 1][np.where(z_prev == 0)[0]]

        if return_time:
            return t
        return idx, X_diverse, X_other

    def generate_recourse(self, x0, k, period=20, best_response=True, interpolate='linear'):
        idx, X_diverse, X_other = self.solve(x0, k, period, best_response)

        recourse_set = []

        if interpolate == 'linear_best':
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(self.data[self.labels == 0])
        
        for i in range(X_diverse.shape[0]):
            recourse_set_i = []
            if interpolate == 'linear_best':
                idx = knn.kneighbors(X_diverse[i].reshape(1, -1), return_distance=False)
            
                for j in range(k):
                    best_x_b = line_search(self.model, x0, X_diverse[i], self.data[self.labels == 0][idx[0][j]], p=2)
                    recourse_set_i.append(best_x_b)
            
                recourse_set += recourse_set_i

            elif interpolate == 'linear':
                best_x_b = line_search(self.model, x0, X_diverse[i], x0, p=2)  
                recourse_set.append(best_x_b)
 
        recourse_set = np.array(recourse_set)

        if interpolate == 'linear_best':
            A, S, d = self.compute_matrix(x0, recourse_set)
            recourse_set = recourse_set[self.quad(S, d, k, cost_diverse=False) == 1]

        return recourse_set, X_diverse, X_other


def line_search(model, x0, x1, x2, p=2):
    best_x_b = None
    best_dist = np.inf

    lambd_list = np.linspace(0, 1, 100)
    for lambd in lambd_list:
        x_b = (1 - lambd) * x1 + lambd * x2
        label = model.predict(x_b)
        if label == 1:
            dist = np.linalg.norm(x0 - x_b, ord=p)
            if dist < best_dist:
                best_x_b = x_b
                best_dist = dist
    
    return best_x_b


def generate_recourse(x0, model, random_state, params=dict()):
    data = params['train_data']
    labels = params['labels']
    X = data[labels == 1]
    k = params['k']

    theta = params['frpd_params']['theta']
    kernel_width = params['frpd_params']['kernel']
    period = params['frpd_params']['period']
    best_response = params['frpd_params']['response']

    quad =  Solver(model, data, labels, theta, kernel_width)
    plans = quad.generate_recourse(x0, k, period, best_response)[0]
    report = dict(feasible=True)

    return plans, report


def quad_recourse(x0, k, model, data, labels, theta, kernel_width):
    quad =  Solver(model, data, labels, theta, kernel_width)
    quad_time = quad.solve(x0, k, return_time=True)

    return quad_time
