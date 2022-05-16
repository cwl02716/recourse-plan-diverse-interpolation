import numpy as np
import torch
import dppy
import math
import matplotlib.pyplot as plt

from libs.frpd.quad import line_search
 
 
def map_inference_dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        # if di2s[selected_item] < epsilon:
            # break
        selected_items.append(selected_item)
    return selected_items
 
 
def map_inference_dpp_sw(kernel_matrix, window_size, max_length, epsilon=1E-10):
    """
    Sliding window version of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param window_size: positive int
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    v = np.zeros((max_length, max_length))
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    window_left_index = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[window_left_index:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        v[k, window_left_index:k] = ci_optimal
        v[k, k] = di_optimal
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[window_left_index:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        if len(selected_items) >= window_size:
            window_left_index += 1
            for ind in range(window_left_index, k + 1):
                t = math.sqrt(v[ind, ind] ** 2 + v[ind, window_left_index - 1] ** 2)
                c = t / v[ind, ind]
                s = v[ind, window_left_index - 1] / v[ind, ind]
                v[ind, ind] = t
                v[ind + 1:k + 1, ind] += s * v[ind + 1:k + 1, window_left_index - 1]
                v[ind + 1:k + 1, ind] /= c
                v[ind + 1:k + 1, window_left_index - 1] *= c
                v[ind + 1:k + 1, window_left_index - 1] -= s * v[ind + 1:k + 1, ind]
                cis[ind, :] += s * cis[window_left_index - 1, :]
                cis[ind, :] /= c
                cis[window_left_index - 1, :] *= c
                cis[window_left_index - 1, :] -= s * cis[ind, :]
            di2s += np.square(cis[window_left_index - 1, :])
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items
 

def generate_recourse(x0, model, random_state, params=dict()):
    data = params['train_data']
    labels = params['labels']
    k = params['k']
    X = data[labels == 1]

    theta = params['frpd_params']['theta']
    kernel_width = params['frpd_params']['kernel']    
    interpolate = params['frpd_params']['interpolate']
    greedy = params['frpd_params']['greedy']

    report = dict(feasible=True)

    A = (X - x0).T / np.linalg.norm(X - x0, axis=1)
    S = A.T @ A
    d = np.linalg.norm(X - x0, axis=1)
    D = np.exp(-d ** 2 / kernel_width ** 2) * np.identity(d.shape[0])
    L = theta * S + (1 - theta) * D

    idx = map_inference_dpp(L, k) if greedy else map_inference_dpp_sw(L, 1, k)
    X_diverse = X[idx, :]

    recourse_set = []
    for i in range(k):
        if interpolate == 'linear':
            best_x_b = line_search(model, x0, X_diverse[i], x0, p=2)
            recourse_set.append(best_x_b)
    
    plans = np.array(recourse_set)

    return plans, report


def dpp_recourse(x0, X, M, gamma=0.5, sigma=2.):
    """dpp recourse.
        map inference for a dpp kernel
            L = gamma * S + (1 - gamma) * exp(-d**2/sigma**2)
 
    Parameters
    ----------
    x0 :
        x0
    X : 
        positive samples
    M: int
        number of items
    gamma :
        weight 
    sigma :
        kernel width
    """
    A = (X - x0).T / np.linalg.norm(X - x0, axis=1)
    S = A.T @ A
    d = np.linalg.norm(X - x0, axis=1)
    D = np.exp(- d**2/sigma**2) * np.identity(d.shape[0])
    L = gamma * S + (1 - gamma) * D
 
    selected_items = map_inference_dpp_sw(L, 1, M)
    print(selected_items)
    return selected_items
 
 
if __name__ == '__main__':
    d = 2
    M = 100
 
    np.random.seed(42)
    np.set_printoptions(suppress=True)
    x0 = np.array([.1, .2])
    X = np.random.randn(M, d)
 
    print("X = ", X)
    gamma = 1
    sigma = 10.
    slt_idx = dpp_recourse(x0, X, 4, gamma=gamma, sigma=sigma)
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[slt_idx] = True
 
    print(mask)
    fig, ax = plt.subplots()
 
    selected_points = X[mask, :]
    print(~mask)
    nonselected_points = X[~mask, :]
    ax.scatter(selected_points[:, 0], selected_points[:, 1], marker='o', color='red')
    ax.scatter(nonselected_points[:, 0], nonselected_points[:, 1], marker='o', color='green')
    ax.scatter(x0[0], x0[1], marker='*', color='blue')
 
    for e in selected_points:
        plt.plot([x0[0], e[0]], [x0[1], e[1]])
 
    ax.set_title(f"$\\gamma = {gamma}, \\sigma = {sigma}$")
    plt.show()
