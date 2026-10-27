"""Validation audit for the frozen observable condition-diversity route."""
from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from cmapss_data import SUBSETS,load_subset
from approaches.condition_normalization import ConditionAwareNormalizer
from approaches.next_experiment_utils import enhanced_regression_metrics
from approaches.training_common import regression_metrics
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'CMAPSSData'; OUT=ROOT/'reports'/'domain_aware_mission';SEED=20261021
def combined():
 f=[]
 for i,s in enumerate(SUBSETS,1):
  x=load_subset(DATA,s)['train'].copy();x['subset']=s;x['source_unit']=x.unit;x['unit']=x.unit+i*10000;f.append(x)
 return pd.concat(f,ignore_index=True).sort_values(['unit','cycle']).reset_index(drop=True)
def main():
 a=pd.read_csv(ROOT/'reports'/'next_experiment'/'validation_predictions_hgb_cap100_normalized.csv'); b=pd.read_csv(ROOT/'reports'/'next_experiment'/'validation_predictions_hgb_cap200_normalized.csv'); p=a[['unit','source_unit','subset','cutoff_cycle','actual_rul','fold','predicted_rul']].rename(columns={'predicted_rul':'cap100_prediction'}).merge(b[['unit','cutoff_cycle','fold','predicted_rul']].rename(columns={'predicted_rul':'cap200_prediction'}),on=['unit','cutoff_cycle','fold'])
 raw=combined();inv=raw.groupby('unit').subset.first().reset_index(); folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(inv,inv.subset,inv.unit)); assigned=[]
 for n,(tr,va) in enumerate(folds,1):
  devunits=set(inv.iloc[tr].unit); vaunits=set(inv.iloc[va].unit); norm=ConditionAwareNormalizer().fit(raw[raw.unit.isin(devunits)]); vf=raw[raw.unit.isin(vaunits)].copy();_,c=norm.transform(vf);vf['cluster']=c
  q=p[p.fold.eq(n)].copy()
  # For each pseudo-censored endpoint, count only clusters observed up to cutoff.
  lookup={(int(u),int(cut)):int(vf[(vf.unit==u)&(vf.cycle<=cut)].cluster.nunique()) for u,cut in zip(q.unit,q.cutoff_cycle)}
  q['observed_condition_count']=[lookup[(int(u),int(c))] for u,c in zip(q.unit,q.cutoff_cycle)];assigned.append(q)
 out=pd.concat(assigned,ignore_index=True);out['route']=np.where(out.observed_condition_count>1,'cap200_multi_condition','cap100_single_condition');out['predicted_rul']=np.where(out.observed_condition_count>1,out.cap200_prediction,out.cap100_prediction);out.to_csv(OUT/'observable_router_validation_predictions.csv',index=False)
 rows=[]
 for s,x in out.groupby('subset'):rows.append({'model':'observable_condition_diversity_router','subset':s,**enhanced_regression_metrics(x.actual_rul,x.predicted_rul,regression_metrics)})
 rows.append({'model':'observable_condition_diversity_router','subset':'Pooled',**enhanced_regression_metrics(out.actual_rul,out.predicted_rul,regression_metrics)})
 pd.DataFrame(rows).to_csv(OUT/'observable_router_validation_metrics.csv',index=False);print(pd.DataFrame(rows)[['subset','rmse','mae','phm08_mean']].to_string(index=False))
if __name__=='__main__':main()
