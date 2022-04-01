import numpy as np

from sklearn.utils import check_random_state

from libs.ar.linear_ar import LinearAR
from libs.explainers.svm import SVMExplainer 

from utils.visualization import visualize_explanations

def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params['train_data']
    ec = params['config']

    explainer = SVMExplainer(
        train_data, model.predict_proba, random_state=rng)

    w, b = explainer.explain_instance(x0,
                                      perturb_radius=ec.perturb_radius * ec.max_distance,
                                      num_samples=ec.num_samples)

    arg = LinearAR(train_data, w, b)
    x_ar = arg.fit_instance(x0)

    # visualize_explanations(model, lines=[(w, b)], x_test=x0, mean_pos=x_ar,
                           # xlim=(-2, 4), ylim=(-4, 7), save=True)
    report = dict(feasible=arg.feasible)

    return x_ar, report
