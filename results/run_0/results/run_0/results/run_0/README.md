# Feasible Recourse Plan via Diverse Interpolation


## Usage

1. Train MLP classifiers

```sh
python train_model.py --clf mlp --data synthesis german sba bank adult --num-proc 16
```

2. Run experiments

* Experiment: cost-DPP and cost-manifold distance trade-off

```sh
python run_expt.py -e 4 --datasets synthesis german sba bank adult -clf mlp --methods frpd_quad frpd_quad_dp frpd_dpp_gr frpd_dpp_ls dice -uc
```

* Experiment: Table Anti-Diversity

```sh
python run_expt.py -e 3 --datasets synthesis german sba bank adult -clf mlp --methods frpd_quad frpd_quad_dp frpd_dpp_gr frpd_dpp_ls dice -uc
```

* Experiment: Run time comparison

```sh
python run_expt.py -e run_time -uc
```
