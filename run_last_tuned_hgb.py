"""One final validation-gated HGB feature/tuning experiment."""
from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import StratifiedGroupKFold
from cmapss_data import SUBSETS, load_subset, add_training_rul
from approaches.condition_normalization import ConditionAwareNormalizer
from approaches.feature_variants import CausalFeatureVariant
from approaches.next_experiment_utils import full_range_endpoint_metadata,endpoint_feature_matrix,enhanced_regression_metrics
from approaches.tabular_models import fit_tabular_model
from approaches.training_common import regression_metrics
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'CMAPSSData';OUT=ROOT/'reports'/'domain_aware_mission';ART=ROOT/'artifacts'/'domain_aware_mission'/'last_tuned_hgb';SEED=20261021
DROP=['sensor_1','sensor_5','sensor_6','sensor_10','sensor_16','sensor_18','sensor_19']
PARAMS={'max_iter':350,'learning_rate':.04,'max_leaf_nodes':31,'min_samples_leaf':15,'l2_regularization':1.5,'early_stopping':False}
def combined():
 f=[]
 for i,s in enumerate(SUBSETS,1):
  x=load_subset(DATA,s)['train'].copy();x['subset']=s;x['source_unit']=x.unit;x['unit']=x.unit+i*10000;f.append(x)
 return pd.concat(f,ignore_index=True).sort_values(['unit','cycle']).reset_index(drop=True)
def transform(dev,target):
 norm=ConditionAwareNormalizer().fit(dev);d,_=norm.transform(dev);t,_=norm.transform(target);v=CausalFeatureVariant(windows=(10,30,50),include_initial_differences=True).fit(d);xd=v.transform(d).drop(columns=[c for c in DROP if c in v.transform(d)],errors='ignore');xt=v.transform(t).drop(columns=[c for c in DROP if c in v.transform(t)],errors='ignore');return norm,v,xd,xt
def main():
 raw=combined();inv=raw.groupby('unit').subset.first().reset_index();folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(inv,inv.subset,inv.unit));outs=[]
 for n,(tr,va) in enumerate(folds,1):
  d=add_training_rul(raw[raw.unit.isin(inv.iloc[tr].unit)],200);vraw=raw[raw.unit.isin(inv.iloc[va].unit)];norm,var,xd,xv=transform(d,vraw);model=fit_tabular_model('hist_gradient_boosting',xd,d.rul.to_numpy(),d.unit.to_numpy(),SEED,PARAMS);meta=full_range_endpoint_metadata(vraw);p=model.predict(endpoint_feature_matrix(vraw,xv,meta));outs.append(meta.assign(fold=n,predicted_rul=p))
 p=pd.concat(outs,ignore_index=True);rows=[]
 for s,x in list(p.groupby('subset'))+[('Pooled',p)]:rows.append({'model':'tuned_hgb_cap200_10_30_50_baseline_delta_sensor_screen','subset':s,**enhanced_regression_metrics(x.actual_rul,x.predicted_rul,regression_metrics)})
 pd.DataFrame(rows).to_csv(OUT/'last_tuned_hgb_validation_metrics.csv',index=False);p.to_csv(OUT/'last_tuned_hgb_validation_predictions.csv',index=False);print(pd.DataFrame(rows)[['subset','rmse','mae','phm08_mean']].to_string(index=False))
if __name__=='__main__':main()
