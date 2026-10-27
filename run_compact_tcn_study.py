"""Bounded compact TCN validation against the global C-MAPSS protocol."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import joblib
from cmapss_data import SUBSETS, SENSOR_COLUMNS, SETTING_COLUMNS, add_training_rul, load_subset
from approaches.condition_normalization import ConditionAwareNormalizer
from approaches.compact_tcn import CompactTCNRegressor
from approaches.next_experiment_utils import full_range_endpoint_metadata, enhanced_regression_metrics
from approaches.training_common import regression_metrics, phm08_contributions

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'CMAPSSData'; OUT=ROOT/'reports'/'domain_aware_mission'; ART=ROOT/'artifacts'/'domain_aware_mission'/'compact_tcn'; SEED=20261021
FEATURES=SETTING_COLUMNS+SENSOR_COLUMNS
def combined():
    frames=[]
    for i,s in enumerate(SUBSETS,1):
        x=load_subset(DATA,s)['train'].copy();x['subset']=s;x['source_unit']=x.unit;x['unit']=x.unit+i*10000;frames.append(x)
    return pd.concat(frames,ignore_index=True).sort_values(['unit','cycle']).reset_index(drop=True)
def windows(frame, endpoints=None, stride=12, size=30):
    chunks=[]; rows=[]
    wanted=None if endpoints is None else {(int(a),int(b)) for a,b in zip(endpoints.unit,endpoints.cutoff_cycle)}
    for unit,g in frame.sort_values(['unit','cycle']).groupby('unit',sort=False):
        arr=g[FEATURES].to_numpy(np.float32); cycles=g.cycle.to_numpy(); labels=g.rul.to_numpy() if 'rul' in g else None
        ends=[]
        if wanted is None: ends=list(range(size-1,len(g),stride))
        else: ends=[i for i,c in enumerate(cycles) if (int(unit),int(c)) in wanted]
        for end in ends:
            start=max(0,end-size+1); z=arr[start:end+1]
            if len(z)<size: z=np.concatenate([np.repeat(z[:1],size-len(z),axis=0),z])
            chunks.append(z); rows.append({'unit':int(unit),'cycle':int(cycles[end]),'actual_rul':float(labels[end]) if labels is not None else np.nan})
    return np.asarray(chunks,np.float32),pd.DataFrame(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True); raw=combined(); inv=raw.groupby('unit').subset.first().reset_index(); folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(inv,inv.subset,inv.unit)); outputs=[]; diagnostics=[]
    for n,(tr,va) in enumerate(folds,1):
        devunits=set(inv.iloc[tr].unit); validunits=set(inv.iloc[va].unit); dev=add_training_rul(raw[raw.unit.isin(devunits)],200); valid=raw[raw.unit.isin(validunits)].copy(); norm=ConditionAwareNormalizer().fit(dev); dn,_=norm.transform(dev); vn,_=norm.transform(valid)
        # Target capping is kept consistent with the selected global cap-200 candidate.
        xn,meta_train=windows(dn,stride=12); yn=meta_train.actual_rul.to_numpy(); endpoints=full_range_endpoint_metadata(valid); xnval,meta_val=windows(vn,endpoints); meta_val=meta_val.merge(endpoints[['unit','cutoff_cycle','subset','actual_rul']],left_on=['unit','cycle'],right_on=['unit','cutoff_cycle'],suffixes=('_window',''))
        model=CompactTCNRegressor(channels=16,epochs=12,batch_size=256,patience=3,seed=SEED+n).fit(xn,yn,xnval,meta_val.actual_rul.to_numpy())
        p=model.predict(xnval); out=meta_val[['unit','subset','cutoff_cycle','actual_rul']].assign(model='compact_multiscale_tcn_attention_cap200',fold=n,predicted_rul=p);out['phm08_contribution']=phm08_contributions(out.actual_rul,p);outputs.append(out);diagnostics.append({'fold':n,'train_sequences':len(xn),'validation_endpoints':len(out),'epochs_completed':model.epochs_completed_,'validation_scaled_mse':model.validation_scaled_mse_})
        # Local PyTorch classes are deliberately saved as state_dict rather
        # than pickled objects, with the preprocessing state beside them.
        import torch
        torch.save({'state_dict':model.model_.state_dict(),'channels':model.channels,'x_mean':model.x_mean_,'x_std':model.x_std_,'y_mean':model.y_mean_,'y_std':model.y_std_,'features':FEATURES,'rul_cap':200,'fold':n},ART/f'fold_{n}_model.pt')
        joblib.dump({'condition_normalizer':norm,'features':FEATURES,'rul_cap':200,'fold':n,'architecture':'compact_tcn_sensor_temporal_attention'},ART/f'fold_{n}_preprocessing.joblib')
    pred=pd.concat(outputs,ignore_index=True);pred.to_csv(OUT/'compact_tcn_validation_predictions.csv',index=False); metrics=[]
    for name,x in list(pred.groupby('subset'))+[('Pooled',pred)]:metrics.append({'model':'compact_multiscale_tcn_attention_cap200','subset':name,**enhanced_regression_metrics(x.actual_rul,x.predicted_rul,regression_metrics)})
    pd.DataFrame(metrics).to_csv(OUT/'compact_tcn_validation_metrics.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'compact_tcn_training_diagnostics.csv',index=False)
    # Artifact is saved after validation only. Official test is deliberately not used to select this architecture.
    (OUT/'compact_tcn_method.md').write_text('# Compact TCN with sensor and temporal attention\n\nThis CPU-bounded architecture uses three dilated residual TCN blocks (dilations 1, 2, 4), channel attention and temporal attention. Each fold fits its condition normalizer and scaling only on development engines. It is inspired by the cited multi-scale attention literature; it is not a reproduction of the published architecture.\n',encoding='utf-8')
    print(pd.DataFrame(metrics)[['subset','rmse','mae','phm08_mean']].to_string(index=False))
if __name__=='__main__':main()
