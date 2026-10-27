"""Evaluate a predeclared observable operating-regime route between frozen models."""
from __future__ import annotations
from pathlib import Path
import joblib, numpy as np, pandas as pd
from cmapss_data import SUBSETS, load_subset
from approaches.next_experiment_utils import enhanced_regression_metrics
from approaches.training_common import regression_metrics

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'CMAPSSData'; OUT=ROOT/'reports'/'domain_aware_mission'
def main():
    # Route definition was fixed before calculation: telemetry that has observed
    # only one operating cluster uses the conservative cap-100 model; telemetry
    # with >1 observed setting cluster uses the cap-200 condition-aware model.
    # It uses no subset filename, RUL label, fault label or future cycle.
    b100=pd.read_csv(ROOT/'reports'/'domain_aware_next'/'global_train_all_test_all'/'official_test_predictions_all_subsets.csv')
    b200=pd.read_csv(ROOT/'reports'/'next_experiment'/'official_test_predictions_all_subsets.csv')
    saved=joblib.load(ROOT/'artifacts'/'next_experiment'/'selected_validation_candidate'/'model.joblib'); norm=saved['condition_normalizer']
    routes=[]
    for subset in SUBSETS:
        telemetry=load_subset(DATA,subset)['test']; _,cluster=norm.transform(telemetry)
        observed=telemetry[['unit','cycle']].copy(); observed['condition_cluster']=cluster
        # A decision at the final available telemetry cycle uses only history
        # received up to that point.
        diversity=observed.groupby('unit').condition_cluster.nunique().rename('observed_condition_count').reset_index()
        diversity['route']=np.where(diversity.observed_condition_count>1,'cap200_multi_condition','cap100_single_condition')
        routes.append(diversity.assign(subset=subset))
    routes=pd.concat(routes,ignore_index=True); routes.to_csv(OUT/'observable_router_assignments.csv',index=False)
    merged=b100[['subset','unit','actual_rul','predicted_rul']].rename(columns={'predicted_rul':'cap100_prediction'}).merge(b200[['subset','unit','predicted_rul']].rename(columns={'predicted_rul':'cap200_prediction'}),on=['subset','unit']).merge(routes,on=['subset','unit'])
    merged['predicted_rul']=np.where(merged.route.eq('cap200_multi_condition'),merged.cap200_prediction,merged.cap100_prediction);merged['model']='observable_condition_diversity_router';merged.to_csv(OUT/'observable_router_historical_predictions.csv',index=False)
    rows=[]
    for subset,x in merged.groupby('subset',sort=True): rows.append({'model':'observable_condition_diversity_router','subset':subset,**enhanced_regression_metrics(x.actual_rul,x.predicted_rul,regression_metrics)})
    rows.append({'model':'observable_condition_diversity_router','subset':'Pooled',**enhanced_regression_metrics(merged.actual_rul,merged.predicted_rul,regression_metrics)})
    pd.DataFrame(rows).to_csv(OUT/'observable_router_historical_metrics.csv',index=False);print(pd.DataFrame(rows)[['subset','rmse','mae','phm08_mean']].to_string(index=False))
if __name__=='__main__':main()
