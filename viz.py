import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

import dice_ml

import networkx as nx

from utils import helpers
from utils.data_transformer import DataTransformer
from expt.common import synthetic_params, load_models


plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=11)    # legend fontsize

df, numerical = helpers.get_dataset("synthesis", params=synthetic_params)
full_dice_data = dice_ml.Data(dataframe=df,
                 continuous_features=numerical,
                 outcome_name='label')
transformer = DataTransformer(full_dice_data)

y = df['label'].to_numpy()
X_df = df.drop('label', axis=1)
X = transformer.transform(X_df).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=42, stratify=y)
idx = np.where(X_train[:, 0] < 0.65)
X_train = X_train[idx]
idx = np.where(X_train[:, 0] > 0.25)
X_train = X_train[idx]
idx = np.where(X_train[:, 1] > 0.25)
X_train = X_train[idx]
# X_train = np.concatenate([np.array([0.55, 0.72]).reshape(1, -1), X_train])
# y_train = np.concatenate([np.array([1]), y_train])
# X_train = np.concatenate([np.array([0.57, 0.76]).reshape(1, -1), X_train])
# y_train = np.concatenate([np.array([1]), y_train])
X_train = np.concatenate([X_train, np.array([0.65, 0.65]).reshape(1, -1)])
X_train = np.concatenate([X_train, np.array([0.5, 0.5]).reshape(1, -1)])
X_train = np.concatenate([X_train, np.array([0.65, 0.55]).reshape(1, -1)])
X_train = np.concatenate([X_train, np.array([0.48, 0.62]).reshape(1, -1)])
X_train = np.concatenate([X_train, np.array([0.43, 0.58]).reshape(1, -1)])
X_train = np.concatenate([X_train, np.array([0.25, 0.35]).reshape(1, -1)])
X_train = np.concatenate([X_train, np.array([0.66, 0.7]).reshape(1, -1)])


model = load_models("synthesis", "mlp", "results/run_0/expt_3")
y_pred = model.predict(X_test)
uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
x0 = uds_X[0]                                          

train_data, labels = X_train, model.predict(X_train)
train_data_ = np.concatenate([x0.reshape(1, -1), train_data])
labels_ = np.concatenate([np.array([0]), labels])

# Build graph 
graph = kneighbors_graph(train_data_, n_neighbors=5, mode='distance', include_self=True, n_jobs=-1)
graph = graph.toarray()
G = nx.from_numpy_array(graph)
p = nx.shortest_path(G, source=0)
print(p[len(graph)-1])

# xmin, xmax = np.min(X_train[:, 0]), np.max(X_train[:, 0])    
xmin, xmax= 0.2, 0.7
# ymin, ymax = np.min(X_train[:, 1]), np.max(X_train[:, 1])    
ymin, ymax = 0.2, 0.8 

fig, ax = plt.subplots()                             
                                                     
xd = np.linspace(xmin, xmax, 1000)                      
yd = np.linspace(ymin, ymax, 1000)                      
X_mesh, Y_mesh = np.meshgrid(xd, yd)                 
inp = torch.tensor([X_mesh, Y_mesh]).permute(1, 2, 0)
out = model.predict(inp)                             
cs = ax.contourf(X_mesh, Y_mesh, out,                
                 cmap="Blues", alpha=0.3, levels=1)  

s = set()
for i in range(graph.shape[0]):
    for j in range(graph.shape[1]):
        if graph[i][j] > 0:
            if labels_[i] == 1 and labels_[j] == 1:
                continue
            s.add(i)
            s.add(j)
            ax.plot([train_data_[i][0], train_data_[j][0]], [train_data_[i][1], train_data_[j][1]], color=(0, 0.1, 0, 0.1))

for i in range(len(graph) - 1, len(graph) - 3, -1):
    path = p[i]
    for j in range(len(path) - 1):
        ax.plot([train_data_[path[j]][0], train_data_[path[j + 1]][0]], [train_data_[path[j]][1], train_data_[path[j + 1]][1]], color='g', linewidth=2.0)

for i in range(len(graph) - 3, len(graph) - 5, -1):
    path = p[i]
    for j in range(len(path) - 1):
        ax.plot([train_data_[path[j]][0], train_data_[path[j + 1]][0]], [train_data_[path[j]][1], train_data_[path[j + 1]][1]], color='r', linewidth=2.0)


s.remove(0)
s = list(s)
data_plot = train_data_[s]
labels_plot = labels_[s]

ax.scatter(x0[0], x0[1], marker="*", color="black", label="Input", s=200)
ax.annotate('$x_0$', (x0 + np.full_like(x0, 0.02)))

ax.scatter(*data_plot[labels_plot==0].T, marker="*", label="Data class 0", s=100, color="orange")
ax.scatter(*data_plot[labels_plot==1][:-4].T, marker="o", label="Data class 1", s=100, color="b")
ax.scatter(*data_plot[labels_plot==1][-2:].T, marker="^", s=100, color="g", label="FRPD-QUAD recourse")
ax.scatter(*data_plot[labels_plot==1][-4:-2].T, marker="^", s=100, color="r", label="FACE recourse")

ax.legend(loc="best")
plt.savefig("frpd_face.pdf", dpi=400)
plt.show()
