"""Endpoint-trained C-MAPSS RUL models based on artificial right-censoring."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from cmapss_data import SUBSETS,load_subset,add_training_rul
from approaches.condition_normalization import ConditionAwareNormalizer
from approaches.feature_variants import CausalFeatureVariant
from approaches.next_experiment_utils import full_range_endpoint_metadata,endpoint_feature_matrix,enhanced_regression_metrics
from approaches.training_common import regression_metrics,phm08_contributions
from approaches.tabular_models import fit_tabular_model

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'CMAPSSData';OUT=ROOT/'reports'/'domain_aware_mission';ART=ROOT/'artifacts'/'domain_aware_mission'/'censoring_aware';SEED=20261022
@dataclass(frozen=True)
class Candidate: name:str; cap:int; family:str='hist_gradient_boosting'; params:dict|None=None
CANDIDATES=(Candidate('endpoint_hgb_cap150',150,params={'max_iter':240,'learning_rate':.05,'max_leaf_nodes':31,'min_samples_leaf':12,'l2_regularization':1.,'early_stopping':False}),Candidate('endpoint_hgb_cap200',200,params={'max_iter':240,'learning_rate':.05,'max_leaf_nodes':31,'min_samples_leaf':12,'l2_regularization':1.,'early_stopping':False}),Candidate('endpoint_hgb_cap250',250,params={'max_iter':240,'learning_rate':.05,'max_leaf_nodes':31,'min_samples_leaf':12,'l2_regularization':1.,'early_stopping':False}),Candidate('endpoint_extratrees_cap200',200,'extra_trees',{'n_estimators':180,'min_samples_leaf':2,'max_features':.75,'max_depth':None,'n_jobs':4}))
def combined():
 f=[]
 for i,s in enumerate(SUBSETS,1):
  x=load_subset(DATA,s)['train'].copy();x['subset']=s;x['source_unit']=x.unit;x['unit']=x.unit+i*10000;f.append(x)
 return pd.concat(f,ignore_index=True).sort_values(['unit','cycle']).reset_index(drop=True)
def training_cutoffs(frame):
 # Thirteen artificial truncation points per engine mimic a censored incoming
 # trajectory and prevent long engines/plateau rows from dominating training.
 return full_range_endpoint_metadata(frame,fractions=(.08,.15,.22,.30,.38,.45,.52,.60,.68,.75,.82,.90,.96))
def transform(dev,target):
 norm=ConditionAwareNormalizer().fit(dev);dn,_=norm.transform(dev);tn,_=norm.transform(target);variant=CausalFeatureVariant(windows=(10,30),include_initial_differences=True).fit(dn);return norm,variant,variant.transform(dn),variant.transform(tn)
def metrics(pred):
 rows=[]
 for s,x in list(pred.groupby('subset'))+[('Pooled',pred)]:rows.append({'subset':s,**enhanced_regression_metrics(x.actual_rul,x.predicted_rul,regression_metrics)})
 return pd.DataFrame(rows)
def official_test(candidate,raw):
 norm,variant,xtrain,xtest=transform(raw, pd.concat([load_subset(DATA,s)['test'].assign(subset=s,source_unit=lambda z:z.unit,unit=lambda z:z.unit+i*10000) for i,s in enumerate(SUBSETS,1)],ignore_index=True).sort_values(['unit','cycle']).reset_index(drop=True))
 md=training_cutoffs(raw);y=np.minimum(md.actual_rul.to_numpy(),candidate.cap);model=fit_tabular_model(candidate.family,endpoint_feature_matrix(raw,xtrain,md),y,md.unit.to_numpy(),SEED,candidate.params)
 testframe=pd.concat([load_subset(DATA,s)['test'].assign(subset=s,source_unit=lambda z:z.unit,unit=lambda z:z.unit+i*10000) for i,s in enumerate(SUBSETS,1)],ignore_index=True).sort_values(['unit','cycle']).reset_index(drop=True)
 endpoints=[]
 for s in SUBSETS:
  d=load_subset(DATA,s);i=SUBSETS.index(s)+1;q=d['test'].sort_values(['unit','cycle']).groupby('unit',sort=True).tail(1)[['unit','cycle']].merge(d['test_rul'],on='unit');q['subset']=s;q['source_unit']=q.unit;q['unit']=q.unit+i*10000;q=q.rename(columns={'cycle':'cutoff_cycle','rul':'actual_rul'});endpoints.append(q)
 ep=pd.concat(endpoints,ignore_index=True);p=model.predict(endpoint_feature_matrix(testframe,xtest,ep));out=ep.assign(model=candidate.name,predicted_rul=p);return norm,variant,model,out
def main():
 OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True);raw=combined();inv=raw.groupby('unit').subset.first().reset_index();folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(inv,inv.subset,inv.unit));records=[]
 for c in CANDIDATES:
  pred=[]
  for n,(tr,va) in enumerate(folds,1):
   dev=raw[raw.unit.isin(inv.iloc[tr].unit)];valid=raw[raw.unit.isin(inv.iloc[va].unit)];norm,var,xd,xv=transform(dev,valid);cut=training_cutoffs(dev);model=fit_tabular_model(c.family,endpoint_feature_matrix(dev,xd,cut),np.minimum(cut.actual_rul.to_numpy(),c.cap),cut.unit.to_numpy(),SEED,c.params);vm=full_range_endpoint_metadata(valid);p=model.predict(endpoint_feature_matrix(valid,xv,vm));pred.append(vm.assign(candidate=c.name,fold=n,predicted_rul=p,phm08_contribution=phm08_contributions(vm.actual_rul,p)))
  p=pd.concat(pred,ignore_index=True);p.to_csv(OUT/f'censoring_aware_validation_predictions_{c.name}.csv',index=False);summary=metrics(p);summary['candidate']=c.name;summary['cap']=c.cap;summary['family']=c.family;records.append(summary)
 summary=pd.concat(records,ignore_index=True);summary.to_csv(OUT/'censoring_aware_validation_comparison.csv',index=False);chosen=summary[summary.subset.eq('Pooled')].sort_values(['phm08_mean','rmse']).iloc[0];candidate=next(c for c in CANDIDATES if c.name==chosen.candidate);norm,var,model,pred=official_test(candidate,raw);pred.to_csv(OUT/'censoring_aware_historical_predictions.csv',index=False);official=metrics(pred);official['candidate']=candidate.name;official['cap']=candidate.cap;official.to_csv(OUT/'censoring_aware_historical_metrics.csv',index=False);joblib.dump({'condition_normalizer':norm,'feature_variant':var,'model':model,'candidate':candidate.__dict__},ART/'selected_endpoint_model.joblib');(OUT/'censoring_aware_method.md').write_text('# Censoring-aware endpoint ensemble\n\nEach run-to-failure training engine is artificially truncated at 13 fixed elapsed-life fractions. The target is the known remaining life at that cutoff, clipped only to the candidate cap. This produces endpoint-shaped training cases that mirror a test engine whose future trajectory is unavailable. Clustering, condition normalization and causal features are fit only on development engines for each fold.\n',encoding='utf-8');print(summary[summary.subset.eq('Pooled')][['candidate','cap','rmse','mae','phm08_mean']].to_string(index=False));print(official[['subset','rmse','mae','phm08_mean']].to_string(index=False))
if __name__=='__main__':main()
