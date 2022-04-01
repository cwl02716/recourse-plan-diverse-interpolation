import argparse

from expt.config import Config


class Expt1(Config):
    __dictpath__ = 'ec.e1'

    all_clfs = ['net0']
    all_datasets = ['synthesis']
    all_methods = ['lime', 'bw_rmpm', 'fr_rmpm', 'mpm', 'quad_rmpm', 'logdet_rmpm']

    rmpm_params = {
        "bw_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
        "fr_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
        "quad_rmpm": {
            "rho_neg": 1.0,
            "rho_pos": 0.0,
        },
        "logdet_rmpm": {
            "rho_neg": 0.5,
            "rho_pos": 0.7,
        },
        "mpm": {
            "rho_neg": 0.5,
            "rho_pos": 0.7,
        },
    }

    kfold = 5
    rho_neg = 0.01
    rho_pos = 0.01

    perturb_sizes = [500, 1000, 2000, 5000, 10000]
    perturb_radius = 0.05
    perturb_std = 1.0
    num_samples = 5000
    max_ins = 500
    num_neighbors = 10
    sigma_neighbors = 0.001
    r_fid = 0.1
    max_distance = 1.0


class Expt2(Config):
    __dictpath__ = 'ec.e2'

    all_clfs = ['net0']
    all_datasets = ['synthesis']
    all_methods = ['lime', 'bw_rmpm', 'fr_rmpm', 'mpm', 'quad_rmpm', 'logdet_rmpm']

    rmpm_params = {
        "bw_rmpm": {
            "rho_neg": 0.1,
            "rho_pos": 5.0,
            "delta": 10.0
        },
        "fr_rmpm": {
            "rho_neg": 0.1,
            "rho_pos": 5.0,
            "delta": 10.0
        },
        "quad_rmpm": {
            "rho_neg": 0.1,
            "rho_pos": 5.0,
            "delta": 10.0
        },
        "logdet_rmpm": {
            "rho_neg": 0.5,
            "rho_pos": 0.7,
            "delta": 10.0
        },
        "mpm": {
            "rho_neg": 0.5,
            "rho_pos": 0.7,
            "delta": 10.0
        },
    }

    kfold = 100
    rho_neg = 0.01
    rho_pos = 0.01
    delta = 10

    perturb_radius = 0.05
    perturb_std = 1.0
    num_samples = 500
    max_ins = 200
    num_neighbors = 10
    sigma_neighbors = 0.001
    r_fid = 0.1
    max_distance = 1.0


class Expt3(Config):
    __dictpath__ = 'ec.e3'

    all_clfs = ['net0']
    all_datasets = ['synthesis']
    all_methods = ['lime-ar', 'svm-ar']

    rmpm_params = {
        "rho_neg": 10.0,
        "rho_pos": 0.0,
    }

    perturb_radius = {
        "synthesis": 0.1,
        "german": 0.2,
        "sba": 0.1,
        "student": 0.7,
    }

    roar_params = {
        'delta_max': 0.2,
    }

    kfold = 5
    num_future = 100
    cross_validate = False 

    perturb_std = 1.0
    num_samples = 1000
    max_ins = 200
    sigma_neighbors = 0.001

    max_distance = 1.0


class Expt4(Config):
    __dictpath__ = 'ec.e4'

    all_clfs = ['net0']
    all_datasets = ['synthesis']
    all_methods = ['lime_roar', 'fr_rmpm_ar']

    rmpm_params = {
        "rho_neg": 0.5,
        "rho_pos": 0.0,
    }


    perturb_radius = {
        "synthesis": 0.1,
        "german": 0.2,
        "sba": 0.1,
        "student": 0.7,
    }

    roar_params = {
        'delta_max': 0.2,
    }

    params_to_vary = {
        'perturb_radius': {
            'default': 0.2,
            'min': 0.4,
            'max': 0.4,
            'step': 0.2,
        },
        'rho_neg': {
            'default': 1.0,
            'min': 0.0,
            'max': 10.0,
            'step': 1.0,
        },
        'delta_max' : {
            'default': 0.05,
            'min': 0.0,
            'max': 0.2,
            'step': 0.02,
        },
        'none': {
            'min': 0.0,
            'max': 0.0,
            'step': 0.1
        }
    }


    kfold = 5
    num_future = 100

    perturb_std = 1.0
    num_samples = 1000
    max_ins = 200
    max_distance = 1.0


class Expt5(Config):
    __dictpath__ = 'ec.e5'

    all_clfs = ['net0']
    all_datasets = ['synthesis']
    all_methods = ['lime_roar', 'fr_rmpm_ar']

    rmpm_params = {
        "rho_neg": 0.5,
        "rho_pos": 0.0,
    }

    roar_params = {
        'delta_max': 0.2,
    }

    perturb_radius = {
        "synthesis": 0.1,
        "german": 0.2,
        "sba": 0.1,
        "student": 0.7,
    }

    params_to_vary = {
        'perturb_radius': {
            'default': 0.2,
            'min': 0.4,
            'max': 0.4,
            'step': 0.2,
        },
        'rho_neg': {
            'default': 1.0,
            'min': 0.0,
            'max': 10.0,
            'step': 1.0,
        },
        'delta_max' : {
            'default': 0.05,
            'min': 0.0,
            'max': 0.2,
            'step': 0.02,
        },
        'none': {
            'min': 0.0,
            'max': 0.0,
            'step': 0.1
        }
    }


    kfold = 5
    num_future = 100

    perturb_std = 1.0
    num_samples = 1000
    max_ins = 200
    max_distance = 1.0


class ExptConfig(Config):
    __dictpath__ = 'ec'

    e1 = Expt1()
    e2 = Expt2()
    e3 = Expt3()
    e4 = Expt4()
    e5 = Expt5()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dump', default='config.yml', type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--mode', default='merge_cls', type=str)

    args = parser.parse_args()
    if args.load is not None:
        ExptConfig.from_file(args.load)
    ExptConfig.to_file(args.dump, mode=args.mode)
