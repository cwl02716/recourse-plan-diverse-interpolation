import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import dice_ml

from libs.frpd import quad, dpp
from libs.dice import dice

from utils import helpers
from utils.data_transformer import DataTransformer
from utils.visualization import visualize_explanations

from expt.common import synthetic_params, clf_map, load_models


synthetic_params['num_samples'] = 50
df, numerical = helpers.get_dataset("synthesis", params=synthetic_params)
full_dice_data = dice_ml.Data(dataframe=df,
                continuous_features=numerical,
                outcome_name='label')
transformer = DataTransformer(full_dice_data)

y = df['label'].to_numpy()
X_df = df.drop('label', axis=1)
X = transformer.transform(X_df).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                    random_state=42, stratify=y)


d = X.shape[1]
clf = clf_map["mlp"]
model = load_models("synthesis", "mlp", "results/run_0/synthesis")

y_pred = model.predict(X_test)
uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
uds_X, uds_y = uds_X[:100], uds_y[:100]

idx = X_train[:, 0] < 0.5 
X_train = X_train[idx]
y_train = model.predict(X_train)

# x0 = uds_X[0]
x0 = np.array([0.5, -0.5])
print(model.predict(x0))
params_frpd = {"train_data": X_train, "labels": y_train,  "k": 2, "frpd_params": {"theta": 1.0, "kernel": 1.0, "period": 10, "response": True}}
plans_frpd, _ = quad.generate_recourse(x0, model, random_state=42, params=params_frpd)


params_dice = {"dataframe": df, "numerical": numerical, "k": 2, "dice_params": {"proximity_weight": 2.0, "diversity_weight": 0.5}}
plans_dice, _ = dice.generate_recourse(x0, model, random_state=42, params=params_dice)
# plans_dice[0] = np.array([-0.21, -0.26])
# plans_frpd[0] = [0.9, -0.1]
# plans_dice[1] = [1.02, 1.17]
print(plans_frpd, plans_dice)
visualize_explanations(model=model, X=X_train, y=y_train, x_test=x0, show=True, N=1000, xlim=(-0.05, 0.8), ylim=(-0.75, 1.5), plans_frpd=plans_frpd, plans_dice=plans_dice, save=True)

fig, ax = plt.subplots()
ax.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=50, color='red', label="Data")
ax.scatter(plans_frpd[:, 0], plans_frpd[:, 1], marker='^', s=50, color='green', label="FRPD-QUAD recourse")
ax.scatter(plans_dice[:, 0], plans_dice[:, 1], marker='^', s=50, color='blue', label="DiCE recourse")
ax.scatter(x0[0], x0[1], marker='*', s=50, color='black', label="Input")

ax.set_xlim(-0.25, 1.5)
ax.set_ylim(-0.75, 1.25)
ax.legend(loc="lower right")
plt.savefig("plot_2D.pdf", dpi=400)

plt.show()
