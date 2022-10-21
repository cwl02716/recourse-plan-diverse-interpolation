import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import dice_ml
import networkx as nx
import gurobipy as grb
from cvxpy import Minimize, Maximize, Problem, Variable

from libs.frpd.dpp import dpp_recourse
from expt.common import synthetic_params, load_models
from utils import helpers
from utils.data_transformer import DataTransformer


# An object oriented max-flow problem.
class Edge:
    """ An undirected, capacity limited edge. """
    def __init__(self, capacity, n1, n2, cost) -> None:
        self.capacity = capacity
        self.cost = cost
        self.flow = Variable(name=f"{n1}_{n2}")

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [0 <= self.flow, self.flow <= self.capacity]

class Node:                                                          
    """ A node with accumulation. """                                
    def __init__(self, accumulation: float = 0.0) -> None:           
        self.accumulation = accumulation                             
        self.edge_flows = []                                         
                                                                     
    # Returns the node's internal constraints.                       
    def constraints(self):                                           
        return [sum(f for f in self.edge_flows) == self.accumulation]


# Load data and load model
df, numerical = helpers.get_dataset("synthesis", params=synthetic_params)
full_dice_data = dice_ml.Data(dataframe=df,
                 continuous_features=numerical,
                 outcome_name='label')
transformer = DataTransformer(full_dice_data)

y = df['label'].to_numpy()
X_df = df.drop('label', axis=1)
X = transformer.transform(X_df).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=42, stratify=y)

model = load_models("synthesis", "mlp", "results/run_0/expt_3")
y_pred = model.predict(X_test)
uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
x0 = uds_X[0]

train_data, labels = X_train, model.predict(X_train)
train_data_ = np.concatenate([x0.reshape(1, -1), train_data])
labels_ = np.concatenate([np.array([0]), labels])


# Select prototypes
gamma = 1
sigma = 10 
slt_idx = dpp_recourse(x0, train_data[labels==1], 3, gamma=gamma, sigma=sigma)[3]
mask = np.zeros(train_data[labels==1].shape[0], dtype=bool)                   
mask[slt_idx] = True                                      

selected_points = train_data[labels==1][mask, :]                              

# Build graph
nbrs = NearestNeighbors(n_neighbors=4).fit(train_data_)
distances, indices = nbrs.kneighbors(train_data_)
graph = nbrs.kneighbors_graph(train_data_)
count_labs = {}
for i in range(indices.shape[0]):
    count_labs[i] = 0
    for j in range(1, indices.shape[1]):
        if labels_[i] == 1 and labels_[indices[i][j]] == 1:
            count_labs[i] += 1

selected = []
for i in range(indices.shape[0]):
    if count_labs[i] != 3:
        selected.append(i)

weighted_graph = np.zeros(graph.shape)
links = grb.tuplelist()
cost = {}
total_edges = []

tau = 1.0

node_count = graph.shape[0]
nodes = [Node() for i in range(node_count)]

origin = 0
destination = []
for i in range(selected_points.shape[0]):              
    idx_l = np.where(train_data_ == selected_points[i])
    destination.append(idx_l[0][0])                    

nodes[0].accumulation = 3   # Set constraint for source node
for i in range(len(destination)):
    nodes[destination[i]].accumulation = -1

checked, checked_idx = {}, {}
idx = 0
for i in range(graph.shape[0]):
    for j in range(1, indices.shape[1]):
        if [train_data_[i][0], train_data_[indices[i][j]][0], train_data_[i][1], train_data_[indices[i][j]][1]] not in total_edges:
            if labels_[i] == 1 and labels_[indices[i][j]] == 1:
                continue
            checked[i] = True
            total_edges.append([train_data_[i][0], train_data_[indices[i][j]][0], train_data_[i][1], train_data_[indices[i][j]][1]])
        if (i, indices[i][j]) not in links:
            links.append((i, indices[i][j]))
            links.append((indices[i][j], i))
            dist = tau * distances[i][j] if (labels_[i] == 0 or labels_[j] == 0) else (1 - tau) * distances[i][j]
            cost[i, indices[i][j]] = dist
            cost[indices[i][j], i] = dist
        weighted_graph[i][indices[i][j]] = distances[i][j]
        weighted_graph[indices[i][j]][i] = distances[i][j]
idx = 0
for i in checked:
    checked_idx[i] = idx
    idx += 1
print(checked, checked_idx)
print(train_data_[list(checked.keys())].shape)
edges = []
for i in range(graph.shape[0]):
    for j in range(1, indices.shape[1]):
        edges.append(Edge(1.0, i, indices[i][j], distances[i][j])) 
        edges[-1].connect(nodes[i], nodes[indices[i][j]])

        edges.append(Edge(1.0, indices[i][j], i, distances[i][j]))
        edges[-1].connect(nodes[indices[i][j]], nodes[i])    
# Construct the problem.                       
constraints = []                               
for o in nodes + edges:                        
    constraints += o.constraints()             
expression = edges[0].cost * edges[0].flow     
for i in range(1, len(edges)):                 
    expression += edges[i].cost * edges[i].flow
# p = Problem(Minimize(expression), constraints) 
# result = p.solve()                             

# res = []
# for variable in p.variables():
    # print(variable.name(), variable.value)
    # if variable.name() == "Source" or variable.name() == "Sink":        
#     if variable.value > 0.1:                                              
#         print("Variable %s: value %s" % (variable.name(), variable.value))
#         i, j = int(variable.name().split('_')[0]), int(variable.name().split('_')[1])
#         res.append([train_data_[i][0], train_data_[j][0], train_data_[i][1], train_data_[j][1]])                                                                    

# print(result)                                                             


# Network flow LP
# model = grb.Model("NF")
# x = model.addVars(links, obj=cost, name="flow")

# origin = 0
# destination = []
# for i in range(selected_points.shape[0]):
#     idx_l = np.where(train_data_ == selected_points[i])
#     destination.append(idx_l[0][0])

# for i in range(graph.shape[0]):
#     model.addConstr(sum(x[i,j] for i,j in links.select(i, '*')) - sum(x[j,i] for j,i in links.select('*',i)) ==(3 if i == origin else -1 if i in destination else 0 ),'node%s_' % i)
#     for i, j in links.select(i, '*'):
#         model.addConstr(x[i,j] >= 0)
#         if labels_[i] == 0 and labels_[j] == 1:
#             model.addConstr(x[i,j] <= 1)

# model.optimize()
# res = []
# res_sum = 0
# if model.status == grb.GRB.Status.OPTIMAL:
    # print('The final solution is:')
    # for i,j in links:
        # if(x[i,j].x > 0):
            # print(i, j, x[i,j].x)
            # res_sum += weighted_graph[i][j]
            # res.append([train_data_[i][0], train_data_[j][0], train_data_[i][1], train_data_[j][1]])

for i in range(len(total_edges)):
    plt.plot([total_edges[i][0], total_edges[i][1]], [total_edges[i][2], total_edges[i][3]], color=(0, 0.1, 0, 0.1))  

# for i in range(len(res)):
#     plt.plot([res[i][0], res[i][1]], [res[i][2], res[i][3]], color='b', linewidth=2.0)


# Plot 2D data
plt.scatter(x0[0], x0[1], marker="o", label="x_0", s=50)
plt.scatter(*train_data[labels==0].T, marker="*", label="Class 0", s=50)
plt.scatter(*train_data[labels==1].T, marker="^", label="Class 1", s=50)
plt.scatter(*selected_points.T, marker="^", label="Prototypes", s=50)
plt.savefig("flow.png", dpi=400)
plt.show()
