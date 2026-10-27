"""Execute the bounded domain-aware C-MAPSS RUL study requested for TELIPORT."""
from __future__ import annotations

import json, time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cmapss_data import SUBSETS, SENSOR_COLUMNS, SETTING_COLUMNS, add_training_rul, load_subset, quality_report
from approaches.condition_normalization import ConditionAwareNormalizer
from approaches.feature_variants import CausalFeatureVariant
from approaches.next_experiment_utils import full_range_endpoint_metadata, endpoint_feature_matrix, enhanced_regression_metrics, add_rul_buckets
from approaches.tabular_models import fit_tabular_model
from approaches.training_common import regression_metrics, phm08_contributions, software_versions

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'CMAPSSData'; OUT=ROOT/'reports'/'domain_aware_mission'; ART=ROOT/'artifacts'/'domain_aware_mission'; SEED=20261021
HGB={'max_iter':160,'learning_rate':.08,'max_leaf_nodes':31,'min_samples_leaf':20,'l2_regularization':1.,'early_stopping':False}

def metric(y,p): return enhanced_regression_metrics(np.asarray(y),np.maximum(np.asarray(p),0),regression_metrics)
def bundle_fit(dev, target, cap, normalized=True, initial=True):
    dev=add_training_rul(dev,cap); norm=ConditionAwareNormalizer().fit(dev) if normalized else None
    d,_=norm.transform(dev) if norm else (dev,None); t,_=norm.transform(target) if norm else (target,None)
    variant=CausalFeatureVariant(windows=(10,30),include_initial_differences=initial).fit(d)
    return norm,variant,variant.transform(d),variant.transform(t),dev
def endpoint(frame):
    f=frame.copy(); f['subset']=f.get('subset','local'); f['source_unit']=f['unit']; return full_range_endpoint_metadata(f)
def preds_for(model, features, frame, meta): return model.predict(endpoint_feature_matrix(frame,features,meta))
def plots(pred, stem):
    figdir=OUT/'figures'; figdir.mkdir(exist_ok=True)
    for group,b in pred.groupby('subset',sort=True):
        plt.figure(figsize=(5,4)); plt.scatter(b.actual_rul,b.predicted_rul,s=13,alpha=.55); lim=max(b.actual_rul.max(),b.predicted_rul.max())+5; plt.plot([0,lim],[0,lim],'k--'); plt.xlabel('Actual RUL');plt.ylabel('Predicted RUL');plt.title(f'{group} {stem}');plt.tight_layout();plt.savefig(figdir/f'{stem}_{group}_actual_vs_pred.png',dpi=150);plt.close()
        plt.figure(figsize=(5,4));plt.scatter(b.actual_rul,b.predicted_rul-b.actual_rul,s=13,alpha=.55);plt.axhline(0,color='k',ls='--');plt.xlabel('Actual RUL');plt.ylabel('Prediction minus actual');plt.title(f'{group} {stem} residual');plt.tight_layout();plt.savefig(figdir/f'{stem}_{group}_residual.png',dpi=150);plt.close()

def audit():
    OUT.mkdir(parents=True,exist_ok=True); ART.mkdir(parents=True,exist_ok=True)
    rows=[]
    for s in SUBSETS:
        obj=load_subset(DATA,s); rows.extend([quality_report(obj['train'],s,'train'),quality_report(obj['test'],s,'test')])
    pd.DataFrame(rows).to_csv(OUT/'audit_data_integrity.csv',index=False)
    b100=pd.read_csv(ROOT/'reports'/'domain_aware_next'/'global_train_all_test_all'/'official_test_predictions_all_subsets.csv')
    b200=pd.read_csv(ROOT/'reports'/'next_experiment'/'official_test_predictions_all_subsets.csv')
    result=[]
    for name,p,cap in [('global_hgb_cap100',b100,100),('global_hgb_cap200',b200,200)]:
        for s,x in p.groupby('subset'):
            result.append({'model':name,'subset':s,'target_view':'raw_actual_rul',**metric(x.actual_rul,x.predicted_rul)})
            result.append({'model':name,'subset':s,'target_view':f'capped_at_{cap}_diagnostic',**metric(np.minimum(x.actual_rul,cap),np.minimum(x.predicted_rul,cap))})
        result.append({'model':name,'subset':'Pooled','target_view':'raw_actual_rul',**metric(p.actual_rul,p.predicted_rul)})
    pd.DataFrame(result).to_csv(OUT/'audit_raw_and_capped_metrics.csv',index=False)
    pd.concat([b100.assign(model='global_hgb_cap100'),b200.assign(model='global_hgb_cap200')]).to_csv(OUT/'audit_historical_predictions.csv',index=False)
    return b100,b200

def specialist_experiments():
    all_summary=[]; test_rows=[]; val_rows=[]
    for subset in SUBSETS:
        data=load_subset(DATA,subset); train=data['train']; engines=np.sort(train.unit.unique()); folds=list(GroupKFold(n_splits=3).split(engines,groups=engines))
        # Candidate cap test is internal development selection.
        cap_scores={100:[],200:[]}
        for cap in cap_scores:
            for fold,(tr,va) in enumerate(folds,1):
                dev=train[train.unit.isin(engines[tr])]; valid=train[train.unit.isin(engines[va])]; use_norm=subset in ('FD002','FD004')
                norm,var,xd,xv,labeled=bundle_fit(dev,valid,cap,use_norm,True); model=fit_tabular_model('hist_gradient_boosting',xd,labeled.rul.to_numpy(),labeled.unit.to_numpy(),SEED,HGB)
                meta=endpoint(valid); p=preds_for(model,xv,valid,meta); rec=meta.assign(subset=subset,candidate=f'specialist_cap{cap}',fold=fold,predicted_rul=p); cap_scores[cap].append(rec)
        score_table=[]
        for cap,recs in cap_scores.items():
            p=pd.concat(recs,ignore_index=True); m=metric(p.actual_rul,p.predicted_rul); score_table.append({'subset':subset,'candidate':f'specialist_cap{cap}','cap':cap,**m}); val_rows.append(p)
        score_df=pd.DataFrame(score_table); selected=int(score_df.sort_values(['phm08_mean','rmse']).iloc[0].cap); all_summary.extend(score_table)
        # refit selected specialist only after internal selection
        norm,var,xt,xte,labeled=bundle_fit(train,data['test'],selected,subset in ('FD002','FD004'),True); model=fit_tabular_model('hist_gradient_boosting',xt,labeled.rul.to_numpy(),labeled.unit.to_numpy(),SEED,HGB)
        test_meta=data['test'].sort_values(['unit','cycle']).groupby('unit',sort=True).tail(1)[['unit','cycle']].merge(data['test_rul'],on='unit').rename(columns={'cycle':'cutoff_cycle','rul':'actual_rul'}); test_meta['subset']=subset; test_meta['source_unit']=test_meta.unit
        p=model.predict(xte.loc[data['test'].sort_values(['unit','cycle']).groupby('unit',sort=True).tail(1).index]); p=np.maximum(p,0)
        out=test_meta.assign(model=f'specialist_hgb_cap{selected}',predicted_rul=p); test_rows.append(out)
        d=ART/f'specialist_{subset}';d.mkdir(parents=True,exist_ok=True);joblib.dump({'normalizer':norm,'feature_variant':var,'model':model,'rul_cap':selected,'subset':subset},d/'model.joblib')
    val=pd.concat(val_rows,ignore_index=True); val.to_csv(OUT/'specialist_validation_predictions.csv',index=False)
    pd.DataFrame(all_summary).to_csv(OUT/'specialist_validation_comparison.csv',index=False)
    test=pd.concat(test_rows,ignore_index=True); summary=[]
    for s,x in test.groupby('subset'): summary.append({'model':x.model.iloc[0],'subset':s,**metric(x.actual_rul,x.predicted_rul)})
    summary.append({'model':'specialists_pooled','subset':'Pooled',**metric(test.actual_rul,test.predicted_rul)})
    pd.DataFrame(summary).to_csv(OUT/'specialist_official_metrics.csv',index=False); test.to_csv(OUT/'specialist_official_predictions.csv',index=False);plots(test,'specialists')
    return pd.DataFrame(all_summary),test

def transfer_fd24():
    """Source pretraining plus target residual adaptation; all fits train only."""
    rows=[]; predictions=[]
    for target in ('FD002','FD004'):
        td=load_subset(DATA,target); source=[]
        for index,s in enumerate(SUBSETS, start=1):
            if s!=target:
                block=load_subset(DATA,s)['train'].copy()
                # Preserve sequence boundaries when source subsets reuse unit IDs.
                block['unit']=block['unit'].astype(int)+index*10000
                source.append(block)
        source=pd.concat(source,ignore_index=True); engines=np.sort(td['train'].unit.unique()); folds=list(GroupKFold(n_splits=3).split(engines,groups=engines))
        for fold,(tr,va) in enumerate(folds,1):
            dev=td['train'][td['train'].unit.isin(engines[tr])]; valid=td['train'][td['train'].unit.isin(engines[va])]
            src=add_training_rul(source,200); devl=add_training_rul(dev,200); norm=ConditionAwareNormalizer().fit(pd.concat([src,devl]))
            srcn,_=norm.transform(src); devn,_=norm.transform(devl); van,_=norm.transform(valid); var=CausalFeatureVariant(windows=(10,30),include_initial_differences=True).fit(pd.concat([srcn,devn])); xs=var.transform(srcn); xd=var.transform(devn); xv=var.transform(van)
            # Keep this transfer baseline laptop-bounded. Features still see
            # complete causal histories, but the source fitting rows are a
            # deterministic engine-balanced sample, capped at 90k records.
            rng=np.random.default_rng(SEED + fold)
            per_engine=max(1, 90000 // src.unit.nunique())
            selected=[]
            for _, positions in src.groupby('unit',sort=False).indices.items():
                positions=np.asarray(positions); selected.extend(rng.choice(positions,size=min(len(positions),per_engine),replace=False).tolist())
            selected=np.asarray(selected,dtype=int)
            source_params={**HGB,'max_iter':70,'max_leaf_nodes':15,'min_samples_leaf':30}
            base=fit_tabular_model('hist_gradient_boosting',xs.iloc[selected],src.rul.to_numpy()[selected],src.unit.to_numpy()[selected],SEED,source_params); residual=fit_tabular_model('hist_gradient_boosting',xd,devl.rul.to_numpy()-base.predict(xd),devl.unit.to_numpy(),SEED,{**HGB,'max_iter':70,'max_leaf_nodes':15,'min_samples_leaf':25})
            meta=endpoint(valid); p=base.predict(endpoint_feature_matrix(valid,xv,meta))+residual.predict(endpoint_feature_matrix(valid,xv,meta)); predictions.append(meta.assign(subset=target,model='source_pretrain_target_residual_adaptation',fold=fold,predicted_rul=np.maximum(p,0)))
    p=pd.concat(predictions,ignore_index=True); p.to_csv(OUT/'transfer_validation_predictions.csv',index=False)
    for s,x in p.groupby('subset'): rows.append({'subset':s,'model':'source_pretrain_target_residual_adaptation',**metric(x.actual_rul,x.predicted_rul)})
    pd.DataFrame(rows).to_csv(OUT/'transfer_validation_metrics.csv',index=False); return pd.DataFrame(rows)

def write_readme(audit100,audit200,specialists,transfer):
    lines=['# Domain Aware C MAPSS Experiment Results','', '## Protocol','All model selection uses engine-grouped validation with causal, pseudo-censored endpoints at 15, 30, 45, 60, 75 and 90 percent observed life. Operating-condition clusters, normalizers, features and models are fit within each development fold. Official test labels are historical audit data because they were inspected in earlier work.','', '## Audit finding','The cap 200 model is not a universal improvement. It reduces high-RUL underprediction in FD002/FD004 but may increase optimistic error in FD001/FD003. `audit_raw_and_capped_metrics.csv` separates raw RUL (business metric) from a cap-matched diagnostic.','', '## Implemented models','- Per-subset HGB specialists: internal comparison of cap 100 and 200; condition normalization is used only for the multi-condition FD002/FD004 subsets.','- Transfer learning proxy: source-domain HGB pretraining followed by a target-domain residual adaptor on FD002 and FD004. This is an implementable tabular transfer baseline rather than a claimed reproduction of the linked neural paper.','- Compact TCN appears in the companion sequence run; it is deliberately CPU-bounded and only retained if grouped validation supports it.','', '## Files','- `audit_data_integrity.csv`: schema, missingness and engine alignment checks.','- `audit_raw_and_capped_metrics.csv`: historical baseline and cap-200 audit.','- `specialist_validation_comparison.csv` and `specialist_official_metrics.csv`: specialist results.','- `transfer_validation_metrics.csv`: source to target transfer validation.','- `figures/`: actual-versus-predicted and residual plots.']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'run_metadata.json').write_text(json.dumps({'seed':SEED,'software_versions':software_versions(),'scope':'Domain-aware C-MAPSS RUL study','test_usage':'historical audit only'},indent=2),encoding='utf-8')

def main():
    a,b=audit()
    # Checkpoint the completed specialist study so a bounded transfer rerun
    # never repeats its already-measured fits.
    if (OUT/'specialist_official_metrics.csv').exists():
        sp=pd.read_csv(OUT/'specialist_validation_comparison.csv'); test=pd.read_csv(OUT/'specialist_official_predictions.csv')
    else:
        sp,test=specialist_experiments()
    tr=transfer_fd24();write_readme(a,b,sp,tr)
    print(pd.read_csv(OUT/'specialist_official_metrics.csv')[['model','subset','rmse','mae','phm08_mean']].to_string(index=False))
if __name__=='__main__': main()
