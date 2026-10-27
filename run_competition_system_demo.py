"""Run the end-to-end sensor-to-work-order demonstration with local MLflow."""
from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from cmapss_data import SUBSETS,SENSOR_COLUMNS,load_subset
from approaches.autoencoder_inputs import build_early_life_windows
from approaches.autoencoder_model import EarlyLifeAutoencoder
from approaches.anomaly_detection import EarlyLifeAnomalyScorer
from triage import RuleBasedTriage

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'CMAPSSData';OUT=ROOT/'reports'/'domain_aware_mission';ART=ROOT/'artifacts'/'domain_aware_mission'/'competition_system'
def tail_windows(frame,size=20,n=3):
 out=[]
 for u,g in frame.sort_values(['unit','cycle']).groupby('unit',sort=True):
  x=g[SENSOR_COLUMNS].to_numpy(np.float32)
  for end in range(max(size-1,len(x)-n),len(x)):
   z=x[end-size+1:end+1]
   if len(z)==size:out.append((int(u),int(g.cycle.iloc[end]),z))
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
 # Early-life telemetry is a normal-reference reconstruction model. C-MAPSS
 # supplies no anomaly labels, so scores are not classification probabilities.
 train=[];test=[]
 for i,s in enumerate(SUBSETS,1):
  d=load_subset(DATA,s);a=d['train'].copy();a['global_engine']=a.unit+i*10000;a['subset']=s;train.append(a);b=d['test'].copy();b['global_engine']=b.unit+i*10000;b['subset']=s;test.append(b)
 train=pd.concat(train,ignore_index=True);test=pd.concat(test,ignore_index=True)
 train_for_anomaly=train.copy(); train_for_anomaly['unit']=train_for_anomaly['global_engine']
 reference,_=build_early_life_windows(train_for_anomaly,early_fraction=.2,window_size=20,sensors=SENSOR_COLUMNS)
 # Deterministic upper bound keeps the demo CPU/memory small.
 reference=reference[np.linspace(0,len(reference)-1,min(12000,len(reference)),dtype=int)]
 anomaly_method='early_life_autoencoder_reconstruction'
 try:
  scorer=EarlyLifeAnomalyScorer(EarlyLifeAutoencoder(window_size=20,hidden_size=32,bottleneck_size=8,epochs=8,batch_size=256,seed=20261021),threshold_quantile=.99).fit(reference)
  scorer.model.save(ART/'early_life_autoencoder.pt'); threshold=float(scorer.threshold)
  def score_windows(values): return scorer.score(values)
 except ImportError as error:
  # Some Windows CPU environments cannot initialise the optional PyTorch DLL.
  # Retain a reproducible unsupervised reconstruction fallback rather than
  # silently omitting the anomaly step.
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler
  anomaly_method='pca_reconstruction_fallback_optional_torch_unavailable'
  scaler=StandardScaler().fit(reference.reshape(len(reference),-1)); pca=PCA(n_components=8,random_state=20261021).fit(scaler.transform(reference.reshape(len(reference),-1)))
  def score_windows(values):
   z=scaler.transform(values.reshape(len(values),-1)); return np.mean((z-pca.inverse_transform(pca.transform(z)))**2,axis=1)
  threshold=float(np.quantile(score_windows(reference),.99)); joblib.dump({'scaler':scaler,'pca':pca,'fallback_reason':str(error),'window_size':20,'sensors':SENSOR_COLUMNS},ART/'pca_anomaly_fallback.joblib')
 joblib.dump({'threshold':threshold,'quantile':.99,'sensors':SENSOR_COLUMNS,'method':anomaly_method},ART/'anomaly_preprocessing.joblib')
 test_for_anomaly=test.copy(); test_for_anomaly['unit']=test_for_anomaly['global_engine']
 stream=tail_windows(test_for_anomaly); windows=np.stack([x[2] for x in stream]);scores=score_windows(windows);flags=scores>threshold
 telemetry=pd.DataFrame({'global_engine':[x[0] for x in stream],'cycle':[x[1] for x in stream],'anomaly_score':scores,'anomaly':flags})
 mapping=test[['subset','unit','global_engine']].drop_duplicates();pred=pd.read_csv(ROOT/'reports'/'next_experiment'/'official_test_predictions_all_subsets.csv')[['subset','unit','predicted_rul']].merge(mapping,on=['subset','unit'])
 telemetry=telemetry.merge(pred[['global_engine','subset','predicted_rul']],on='global_engine',how='left',validate='many_to_one')
 triage=RuleBasedTriage();decisions=[]
 for r in telemetry.sort_values(['global_engine','cycle']).itertuples(index=False):decisions.append(triage.assess(engine_id=int(r.global_engine),cycle=int(r.cycle),predicted_rul=float(r.predicted_rul),anomaly=bool(r.anomaly),anomaly_score=float(r.anomaly_score),anomaly_threshold=threshold,data_quality='good',model_version='global_hgb_cap200_historical_audit',interval_width=None))
 d=pd.DataFrame(decisions);d.to_csv(OUT/'competition_triage_decisions.csv',index=False);orders=pd.DataFrame(triage.work_orders);orders.to_csv(OUT/'competition_work_orders.csv',index=False);(OUT/'competition_work_orders.json').write_text(json.dumps(triage.work_orders,indent=2),encoding='utf-8')
 metrics=pd.read_csv(OUT/'audit_raw_and_capped_metrics.csv');pooled=metrics[(metrics.model=='global_hgb_cap200')&(metrics.subset=='Pooled')&(metrics.target_view=='raw_actual_rul')].iloc[0]
 import mlflow
 mlflow.set_tracking_uri((ROOT/'mlruns').as_uri());mlflow.set_experiment('tel iport domain aware cmapss')
 with mlflow.start_run(run_name='competition_system_global_hgb_cap200'):
  mlflow.log_params({'rul_model':'global_hgb_cap200','condition_normalization':'kmeans_settings_train_fitted','anomaly_detector':anomaly_method,'anomaly_labels_available':False,'triage':'deterministic_rule_based'})
  mlflow.log_metrics({'pooled_historical_rmse':float(pooled.rmse),'pooled_historical_mae':float(pooled.mae),'pooled_historical_phm08_mean':float(pooled.phm08_mean),'anomaly_reference_windows':len(reference),'anomaly_rate_stream':float(np.mean(flags)),'work_orders':len(orders)})
  mlflow.log_artifact(str(OUT/'competition_work_orders.csv'));mlflow.log_artifact(str(ART/'anomaly_preprocessing.joblib'))
 (OUT/'competition_system_metadata.json').write_text(json.dumps({'pipeline':'Sensor data -> training-fitted preprocessing -> operating condition identification -> RUL prediction -> normal-reference anomaly score -> deterministic triage -> mock work order -> local MLflow','anomaly_note':'C-MAPSS has no valid anomaly ground truth; scores are reconstruction errors, not supervised anomaly classes.','anomaly_method':anomaly_method,'anomaly_threshold':threshold,'reference_windows':int(len(reference)),'stream_windows':int(len(windows)),'work_orders':int(len(orders))},indent=2),encoding='utf-8')
 print(f'work_orders={len(orders)} anomaly_rate={np.mean(flags):.4f} mlruns={ROOT / "mlruns"}')
if __name__=='__main__':main()
