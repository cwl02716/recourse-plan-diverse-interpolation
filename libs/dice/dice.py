import pandas as pd

import dice_ml

from libs.dice.dice_wrapper import DicePyTorchWrapper


def generate_recourse(x0, model, random_state, params=dict()):
    df = params['dataframe']
    numerical = params['numerical']
    k = params['k']

    full_dice_data = dice_ml.Data(dataframe=df,
                                  continuous_features=numerical,
                                  outcome_name='label')
    dice_model = dice_ml.Model(
        model=model, backend='PYT')
    dice = DicePyTorchWrapper(full_dice_data, dice_model)      

    df = df.drop(columns=['label'])
    keys = df.columns
    d = {}
    for i in range(len(x0)):
        d[keys[i]] = [x0[i]]
    
    plans = dice.generate_counterfactuals(pd.DataFrame.from_dict(d), total_CFs=k,
                                          desired_class="opposite",
                                          posthoc_sparsity_param=None,
                                          proximity_weight=0.5,
                                          diversity_weight=1.0) 
    
    report = dict(feasible=True)
    
    return plans.final_cfs_df.drop(columns=['label']).to_numpy(), report
