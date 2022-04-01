import os
import numpy as np
import pandas as pd
import joblib
import torch
import sklearn
from collections import defaultdict, namedtuple

from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from utils import helpers
from utils.transformer import get_transformer
from utils.visualization import visualize_explanations
from expt.common import synthetic_params, clf_map
from libs.explainers.clime import CLime
from libs.explainers.lime_wrapper import LimeWrapper
from libs.explainers.svm import SVMExplainer
from utils.funcs import compute_robustness, \
    compute_max_distance, compute_max_shift
from rmpm.explainer import RMPMExplainer

Results = namedtuple("Results", ["rob", "fid", "quad_neg", "bw_neg", "fr_neg",
                                 "quad_pos", "bw_pos", "fr_pos", "mean_neg", "mean_pos"])
Results.__new__.__defaults__ = (0,) * len(Results._fields)


def load_shifted_model(clf, d, dname, cname, kfold):
    models = helpers.pload(f"{cname}_{dname}_{kfold}.pickle", "checkpoints/")
    return models


def compute_counterfactual_fidelity(x0, line, shifted_models):
    epsilon = 0.0
    w, b = line
    x_proj = x0 - min(0, np.dot(w, x0) + b - epsilon) * w / (np.linalg.norm(w) ** 2)
    preds = []
    for model in shifted_models:
        pred = model.predict(x_proj.reshape(1, -1))
        preds.append(pred)
    preds = np.array(preds)
    return preds


def _run_svm(idx, x0, model, shifted_models, train_data, method, ec, random_state, logger):
    logger.info("Running method %s with instance %d...", method, idx)

    explainer = SVMExplainer(
        train_data, model.predict_proba, random_state=random_state)

    exp = explainer.explain_instance(x0,
                                     perturb_radius=ec.perturb_radius * ec.max_distance,
                                     num_samples=ec.num_samples)

    fid = compute_counterfactual_fidelity(
        x0, exp, shifted_models)
    logger.info("done method %s, instance %d!", method, idx)
    return Results(0, fid)


def _run_lime(idx, x0, model, shifted_models, train_data, method, ec, random_state, logger):
    logger.info("Running method %s with instance %d...", method, idx)
    explainer = LimeWrapper(train_data, class_names=['0', '1'],
                            discretize_continuous=False, random_state=random_state)

    exp = explainer.explain_instance(x0, model.predict_proba,
                                     num_samples=ec.num_samples)

    fid = compute_counterfactual_fidelity(
        x0, exp, shifted_models)
    logger.info("done method %s, instance %d!", method, idx)
    return Results(0, fid)


def _run_clime(idx, x0, model, shifted_models, train_data, method, ec, random_state, logger):
    logger.info("Running method %s with instance %d...", method, idx)

    explainer = CLime(train_data, class_names=['0', '1'],
                      discretize_continuous=False, random_state=random_state)

    exp = explainer.explain_instance(x0, model.predict_proba,
                                     perturbation_std=1.0,
                                     num_samples=ec.num_samples)

    fid = compute_counterfactual_fidelity(
        x0, exp, shifted_models)
    logger.info("Done method %s, instance %d!", method, idx)
    return Results(0, fid)


def _run_rmpm(idx, x0, model, shifted_models, train_data, method, ec, random_state, logger):
    logger.info("Running method %s with instance %d...", method, idx)
    explainer = RMPMExplainer(
        train_data, model.predict_proba, random_state=random_state)

    x0_label = model.predict(x0)
    rho_neg = ec.rmpm_params[method]['rho_neg']
    rho_pos = ec.rmpm_params[method]['rho_pos']
    if (method == 'fr_rmpm' or method == 'bw_rmpm' or method == 'quad_rmpm') and x0_label == 0:
        rho_neg, rho_pos = rho_pos, rho_neg
    # rho_neg = rho_pos = 'auto'
    exp = explainer.explain_instance(x0, perturb_radius=ec.perturb_radius * ec.max_distance,
                                     rho_neg=rho_neg, rho_pos=rho_pos,
                                     method=method, num_samples=ec.num_samples)

    fid = compute_counterfactual_fidelity(
        x0, exp, shifted_models)
    logger.info("Done method %s, instance %d!", method, idx)
    return Results(0, fid)


def run(ec, wdir, dname, cname, mname,
        num_proc, seed, logger):
    logger.info("Running dataset: %s, classifier: %s, method: %s...",
                dname, cname, mname)
    random_state = check_random_state(None)
    df, _ = helpers.get_dataset(dname, params=synthetic_params)
    y = df['label'].to_numpy()
    X_df = df.drop('label', axis=1)
    transformer = get_transformer(dname)
    X = transformer.transform(X_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)

    ec.max_distance = compute_max_distance(X_train)

    d = X.shape[1]

    clf = clf_map[cname]
    shifted_models = load_shifted_model(clf, d, dname, cname, ec.kfold)

    path = f'{cname}_{dname}_0.pickle'
    model = helpers.pload(path, 'checkpoints')

    method = method_map[mname]

    d = X_test.shape[1]
    res = defaultdict(dict)

    X_all = np.vstack([X_test, X_train])
    y_all = np.concatenate([y_test, y_train])
    y_pred = model.predict(X_all)
    uds_X, uds_y = X_all[y_pred == 0], y_all[y_pred == 0]
    uds_X, uds_y = uds_X[:ec.max_ins], uds_y[:ec.max_ins]

    jobs_args = []
    for idx, x0 in enumerate(uds_X):
        x_neighbors = x0 + \
            random_state.randn(ec.num_neighbors, d) * ec.sigma_neighbors
        x_neighbors = np.vstack([x0, x_neighbors])

        # run
        jobs_args.append((idx, x0, model, shifted_models, X_train, mname,
                          ec, random_state, logger))

    # fix epsilon, varying k
    rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(method)(
        *jobs_args[i]) for i in range(len(jobs_args)))

    fidelity = []

    for r in rets:
        fidelity.append(r.fid)

    fidelity = np.hstack(fidelity)

    helpers.pdump(fidelity,
                  f'{cname}_{dname}_{mname}.pickle', wdir)

    logger.info("Done dataset: %s, classifier: %s, method: %s!",
                dname, cname, mname)


method_map = {
    "lime": _run_lime,
    "clime": _run_clime,
    "svm": _run_svm,
    "bw_rmpm": _run_rmpm,
    "quad_rmpm": _run_rmpm,
    "fr_rmpm": _run_rmpm,
    "quad_rmpm": _run_rmpm,
    "logdet_rmpm": _run_rmpm,
}


def plot_2(ec, wdir, cname, dname, methods):
    method_name = {
        "bw_rmpm": "BW-RMPM",
        "fr_rmpm": "FR-RMPM",
        "clime": "CLIME",
        "SVM": "SVM",
        "lime": "LIME",
        "mpm": "MPM",
        "quad_rmpm": "QUAD-RMPM",
        "logdet_rmpm": "LOGDET-RMPM",
        "dba": "DBA"
    }
    metric_name = {
        "fid": "Local Fidelity",
        "rob": "Robustness",
    }

    res = {"method": [], "cfid_mean": [], "cfid_std": []}
    for mname in methods:
        fid = helpers.pload(
            f'{cname}_{dname}_{mname}.pickle', wdir)
        res['method'].append(mname)
        avg_fid = np.mean(fid, axis=1)
        res['cfid_mean'].append(np.mean(avg_fid))
        res['cfid_std'].append(np.std(avg_fid))

    df = pd.DataFrame(res)
    print(df)
    filepath = os.path.join(wdir, f"{cname}_{dname}.csv")
    df.to_csv(filepath, index=False)


def plot_3(ec, wdir, cname, datasets, methods):
    method_name = {
        "bw_rmpm": "BW-RMPM",
        "fr_rmpm": "FR-RMPM",
        "clime": "CLIME",
        "svm": "SVM",
        "lime": "LIME",
        "mpm": "MPM",
        "quad_rmpm": "QUAD-RMPM",
        "logdet_rmpm": "LOGDET-RMPM",
        "dba": "DBA"
    }
    metric_name = {
        "fid": "Local Fidelity",
        "rob": "Robustness",
    }

    res = defaultdict(list)
    for mname in methods:
        res['method'].append(method_name[mname])
        for dname in datasets:
            fid = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)
            avg_fid = np.mean(fid, axis=1)
            print(avg_fid)
            res[f'{dname}-cfid-mean'].append(np.mean(avg_fid))
            res[f'{dname}-cfid-std'].append(np.std(avg_fid))

    df = pd.DataFrame(res)
    print(df)
    filepath = os.path.join(wdir, f"{cname}.csv")
    df.to_csv(filepath, index=False, float_format='%.2f')


def run_expt_2(ec, wdir, datasets, classifiers, methods,
               num_proc=4, plot_only=False, seed=None, logger=None):
    logger.info("Running ept 1...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.e2.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.e2.all_clfs

    if methods is None or len(methods) == 0:
        methods = ec.e2.all_methods

    for cname in classifiers:
        for dname in datasets:
            if not plot_only:
                for mname in methods:
                    run(ec.e2, wdir, dname, cname, mname,
                        num_proc, seed, logger)
            plot_2(ec.e2, wdir, cname, dname, methods)
        plot_3(ec.e2, wdir, cname, datasets, methods)

    logger.info("Done ept 1.")
