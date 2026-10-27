"""Create the final measured domain-aware report, OPP summary and PDF."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from approaches.next_experiment_utils import enhanced_regression_metrics
from approaches.training_common import regression_metrics

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'reports'/'domain_aware_mission'; PDF=OUT/'Domain_Aware_CMAPSS_Final_Report.pdf'
def m(y,p): return enhanced_regression_metrics(np.asarray(y),np.asarray(p),regression_metrics)
def table(data,widths=None):
 t=Table(data,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17365D')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7.5),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#D9E2F3')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F6F8FC')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));return t
def fmt(x):return f'{float(x):.3f}'
def main():
 audit=pd.read_csv(OUT/'audit_raw_and_capped_metrics.csv');b100=pd.read_csv(ROOT/'reports'/'domain_aware_next'/'global_train_all_test_all'/'official_test_predictions_all_subsets.csv');b200=pd.read_csv(ROOT/'reports'/'next_experiment'/'official_test_predictions_all_subsets.csv'); specialist=pd.read_csv(OUT/'specialist_official_metrics.csv'); transfer=pd.read_csv(OUT/'transfer_validation_metrics.csv');tcn=pd.read_csv(OUT/'compact_tcn_validation_metrics.csv');routev=pd.read_csv(OUT/'observable_router_validation_metrics.csv');routeh=pd.read_csv(OUT/'observable_router_historical_metrics.csv');tuned=pd.read_csv(OUT/'last_tuned_hgb_validation_metrics.csv');meta=json.loads((OUT/'competition_system_metadata.json').read_text())
 # Critical range is scored on raw RUL targets, not a redefinition of error.
 critical=[]
 for name,p in [('global_hgb_cap100',b100),('global_hgb_cap200',b200)]:
  for subset,x in list(p.groupby('subset'))+[('Pooled',p)]:
   x=x[x.actual_rul.between(0,30)]
   if len(x):critical.append({'model':name,'subset':subset,'rul_range':'0_to_30','cases':len(x),**m(x.actual_rul,x.predicted_rul)})
 pd.DataFrame(critical).to_csv(OUT/'critical_0_to_30_metrics.csv',index=False)
 comp=[]
 for model in ['global_hgb_cap100','global_hgb_cap200']:
  x=audit[(audit.model==model)&(audit.subset=='Pooled')&(audit.target_view=='raw_actual_rul')].iloc[0];comp.append({'model':model,'scope':'historical official audit','selection_status':'cap200 was validation-selected; cap100 benchmark','rmse':x.rmse,'mae':x.mae,'r2':x.r2,'phm08_mean':x.phm08_mean,'bias':x.bias_predicted_minus_actual})
 x=specialist[specialist.subset=='Pooled'].iloc[0];comp.append({'model':'per_subset_specialists','scope':'historical official audit','selection_status':'rejected worse pooled result','rmse':x.rmse,'mae':x.mae,'r2':x.r2,'phm08_mean':x.phm08_mean,'bias':x.bias_predicted_minus_actual})
 x=routeh[routeh.subset=='Pooled'].iloc[0];comp.append({'model':'observable_router','scope':'historical official audit','selection_status':'diagnostic only: failed validation gate','rmse':x.rmse,'mae':x.mae,'r2':x.r2,'phm08_mean':x.phm08_mean,'bias':x.bias_predicted_minus_actual})
 x=tuned[tuned.subset=='Pooled'].iloc[0];comp.append({'model':'tuned_hgb_cap200','scope':'grouped validation','selection_status':'RMSE challenger; PHM08 gate failed','rmse':x.rmse,'mae':x.mae,'r2':x.r2,'phm08_mean':x.phm08_mean,'bias':x.bias_predicted_minus_actual})
 pd.DataFrame(comp).to_csv(OUT/'final_model_comparison.csv',index=False)
 # Figures used in both the PDF and dashboard report directory.
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 plt.figure(figsize=(7,4));x=pd.DataFrame(comp);plt.bar(x.model,x.rmse,color=['#A7A7A7','#1F77B4','#FF9F43','#8E44AD','#2CA02C']);plt.ylabel('RMSE by reported evaluation scope');plt.xticks(rotation=15,ha='right');plt.tight_layout();plt.savefig(figdir/'final_pooled_rmse_comparison.png',dpi=180);plt.close()
 plt.figure(figsize=(6,5));plt.scatter(b200.actual_rul,b200.predicted_rul,s=15,alpha=.55,c='#1F77B4');lim=max(b200.actual_rul.max(),b200.predicted_rul.max())+10;plt.plot([0,lim],[0,lim],'k--');plt.xlim(0,lim);plt.ylim(0,lim);plt.xlabel('Actual official RUL');plt.ylabel('Global HGB cap-200 prediction');plt.title('Historical official endpoint audit');plt.tight_layout();plt.savefig(figdir/'final_cap200_actual_vs_predicted.png',dpi=180);plt.close()
 tail=[]
 for s,x in b100.groupby('subset'):
  tail.append({'subset':s,'engines':len(x),'actual_rul_over_100_percent':100*(x.actual_rul>100).mean(),'actual_rul_over_150_percent':100*(x.actual_rul>150).mean(),'cap100_bias':m(x.actual_rul,x.predicted_rul)['bias_predicted_minus_actual'],'cap200_bias':m(b200[b200.subset==s].actual_rul,b200[b200.subset==s].predicted_rul)['bias_predicted_minus_actual']})
 pd.DataFrame(tail).to_csv(OUT/'cap_effect_target_tail_and_bias.csv',index=False)
 cap200=audit[(audit.model=='global_hgb_cap200')&(audit.target_view=='raw_actual_rul')].copy();cap100=audit[(audit.model=='global_hgb_cap100')&(audit.target_view=='raw_actual_rul')].copy()
 opp=f'''# OPP Summary Domain Aware C MAPSS Predictive Maintenance

## One line proposal
Use a globally trained, operating-condition-normalized HGB RUL predictor with causal sensor features and explainable triage, then deploy it with model monitoring and a normal-reference anomaly score.

## Business problem
The system estimates Remaining Useful Life from multivariate turbofan telemetry so maintenance teams can prioritise engines before risk becomes critical. NASA C-MAPSS is a simulation benchmark. Results here are not certification evidence for aircraft maintenance.

## What was implemented
1. Engine-aligned data audit across FD001–FD004.
2. KMeans operating-condition identification from three observable settings, fitted only on development training rows.
3. Per-condition sensor normalization, causal 10/30-cycle means, standard deviations and trends, plus early-life baseline differences.
4. Global cap-100 and cap-200 HGB comparison, per-subset specialists, a cross-condition pretrain/residual-adaptation transfer baseline, a compact attention TCN, and an observable condition-diversity router.
5. Sensor stream to RUL prediction to normal-reference anomaly score to deterministic triage to idempotent mock work order, logged in local MLflow.

## Measured recommendation
Use **global condition-normalized HGB with cap 200** for this benchmark. Its historical official pooled RMSE is **21.455 cycles**, MAE **14.622**, R2 **0.824**, and PHM08 mean **31.465**. It improves the cap-100 pooled RMSE from **28.737** by **25.3%**. It specifically improves FD002 from 31.128 to 21.738 and FD004 from 33.391 to 22.630.

## USP
- Handles multi-condition fleets without needing a subset filename at deployment.
- Uses training-fitted operating regimes and causal features, avoiding future leakage.
- Carries prediction into an explainable operational workflow, including severity, trigger reason and recommended action.
- Separates RUL prediction from unsupervised reconstruction-error anomaly monitoring because C-MAPSS has no anomaly labels.

## Important tradeoff
Cap 200 is not universally better: historical FD001 RMSE moves from 16.292 to 18.037 and FD003 from 17.273 to 20.863. The reason is target-range mismatch: cap 100 creates conservative high-RUL predictions, especially on FD002/FD004, while cap 200 reduces that underprediction but adds positive bias on the single-condition subsets. Raw RUL remains the business metric; cap-matched values are diagnostics only.

## Research results retained as evidence
- Specialists: pooled official RMSE 22.206, rejected.
- Transfer pretrain plus target residual adaptation: validation RMSE 29.342 on FD002 and 39.319 on FD004, rejected.
- Compact dilated TCN with sensor/time attention: validation pooled RMSE 43.615, rejected.
- Observable router: historical official RMSE 20.797, but validation RMSE 42.781, not promoted.
- Final tuned HGB: validation RMSE 33.401 versus 34.323 for the prior cap-200 HGB, but PHM08 mean rose from 254901 to 292865; retained as a challenger only.

## Operational demo
The demo processed 2,109 tail telemetry windows using 12,000 early-life normal-reference windows and generated 228 idempotent mock work orders. Local MLflow artifacts record parameters, metrics and work-order outputs. In this Windows environment the optional PyTorch DLL failed to initialise for the autoencoder; the persisted demo records a PCA reconstruction fallback rather than hiding it.
'''
 (OUT/'OPP_SUMMARY.md').write_text(opp,encoding='utf-8')
 report=f'''# Domain Aware C MAPSS Predictive Maintenance Final Report

## Executive conclusion
The recommended benchmark model is the global condition-normalized Histogram Gradient Boosting model trained with a 200-cycle RUL cap. It was selected from training-only, engine-grouped validation in the prior cap study and then audited against historically inspected official endpoints. It reduces pooled historical official RMSE from 28.737 to 21.455 cycles. It is a measured improvement for multi-condition FD002 and FD004, while sacrificing FD001 and FD003 performance. The recommendation is therefore a global fleet choice with a clearly recorded operating-domain tradeoff.

## Audit and metric definitions
The audit checked the expected 26-column schema, engine/cycle uniqueness, numeric validity, unit/cycle ordering and test-RUL alignment. All eight train/test files had zero missing cells, nonfinite values, duplicate unit-cycle records, cycle gaps and ordering faults. Test RUL labels are aligned to sorted engine IDs by the loader.

**Raw-target metric** means the supplied official RUL target and unmodified prediction. This is the operational result. **Capped-target diagnostic** clips both target and prediction to the stated training cap; it answers how the model performs on its capped representation and must not replace the raw target result. PHM08 uses `d = prediction - actual`: `exp(-d/13)-1` when d < 0 and `exp(d/10)-1` otherwise. Overprediction is therefore costlier per cycle.

## Leakage controls
Engines are kept intact in all validation folds. Endpoint features only use telemetry at or before each pseudo-censored cutoff. KMeans settings clustering, condition-specific sensor moments, sensor screening, standardization and all fitted models are learned from development engines only. No true test RUL, hidden fault label or subset filename is used to choose a deployment route.

## Result summary
| Model | Scope | RMSE | MAE | R2 | PHM08 mean | Decision |
|---|---:|---:|---:|---:|---:|---|
'''+ '\n'.join(f"| {r.model} | {r.scope} | {r.rmse:.3f} | {r.mae:.3f} | {r.r2:.3f} | {r.phm08_mean:.3f} | {r.selection_status} |" for r in pd.DataFrame(comp).itertuples())+'''\n\n## Why cap 200 helps selectively
FD002 and FD004 contain more high-RUL endpoints: their cap-100 models have strongly negative raw-target bias. Raising the cap lets the model represent those values and improves their RMSE. FD001 and FD003 do not receive the same benefit; their residual bias becomes optimistic. This is why a cap-200 improvement cannot be called universal. The output `cap_effect_target_tail_and_bias.csv` contains the supporting tail and bias counts.

## Specialist and research experiments
Per-subset specialists were genuinely trained and assessed. They selected cap 200 in the full-range grouped protocol but did not beat the global model in pooled historical audit. The source-pretrain plus target-residual transfer model was trained as a computationally bounded tabular transfer baseline; it did not validate. The compact TCN contains dilated residual blocks at rates 1, 2 and 4 plus sensor and temporal attention. It was deliberately kept small and its grouped validation result was much worse, so it is retained for reproducibility and not promoted. A final tuned HGB improved validation RMSE, but its higher PHM08 mean prevented promotion. The dual-scale Transformer was not run because the simpler attention TCN did not clear the validation gate.

## Competition system
Sensor data feeds training-fitted preprocessing and operating-condition identification, then the recommended RUL predictor. A normal-reference reconstruction score runs separately. Deterministic triage converts prediction, score, threshold, data quality and persistence into a severity and recommended action. Each work order records asset, predicted RUL, anomaly score, severity and triggering rule. C-MAPSS has no supervised anomaly labels, so the score is not claimed as anomaly classification. Local MLflow captures the demo run.

## Final recommendation
Deploy the global cap-200 HGB pipeline as the current benchmark candidate, with interval calibration and condition-drift monitoring before any live trial. Keep the cap-100 scorecard for FD001/FD003 comparison. Do not promote specialists, transfer adaptation, the compact TCN or the observable router because their validation evidence does not support it.\n'''
 (OUT/'FINAL_DOMAIN_AWARE_REPORT.md').write_text(report,encoding='utf-8')
 # PDF
 styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='Small',parent=styles['BodyText'],fontSize=8.5,leading=11));styles['Title'].fontName='Helvetica-Bold';styles['Title'].textColor=colors.black
 story=[Paragraph('Domain Aware C MAPSS Predictive Maintenance',styles['Title']),Spacer(1,8),Paragraph('TELIPORT Case Study Final Measured Report',styles['Heading2']),Paragraph('Purpose: document the audit, experiments, operational prototype and evidence-based recommendation for an OPP presentation.',styles['BodyText']),Spacer(1,10),Paragraph('Executive conclusion',styles['Heading1']),Paragraph('The recommended benchmark candidate is a global condition-normalized HGB model with a 200-cycle training cap. It lowers the pooled historical official-test RMSE from 28.737 to 21.455 cycles, while the evidence also records its FD001 and FD003 regression. All official test scores are a historical audit because those labels had already been inspected before this study.',styles['BodyText']),Spacer(1,8),table([['Model','RMSE','MAE','R2','PHM08','Decision']]+[[r.model,fmt(r.rmse),fmt(r.mae),fmt(r.r2),fmt(r.phm08_mean),r.selection_status] for r in pd.DataFrame(comp).itertuples()],[3.1*cm,1.5*cm,1.5*cm,1.3*cm,1.5*cm,7*cm]),Spacer(1,10),Image(str(figdir/'final_pooled_rmse_comparison.png'),width=15*cm,height=8.6*cm),PageBreak()]
 story += [Paragraph('Audit and evaluation protocol',styles['Heading1']),Paragraph('All 8 C-MAPSS train/test files passed schema, completeness, numerical-integrity, engine-cycle uniqueness and alignment checks. Development uses engine-grouped validation and six causal pseudo-censored endpoints per held-out engine: 15%, 30%, 45%, 60%, 75% and 90% observed life. Every fitted transform is learned only on development engines.',styles['BodyText']),Spacer(1,7),Paragraph('Metrics',styles['Heading2']),Paragraph('RMSE, MAE, R2, PHM08 mean and sum, bias, optimistic prediction rate, and within 5/10/15 cycle rates are saved in the CSV artifacts. Raw target metrics use official RUL without clipping. Capped-target metrics clip both target and prediction to the cap and are diagnostic only.',styles['BodyText']),Spacer(1,7),Paragraph('Cap finding',styles['Heading2']),Paragraph('A cap of 200 allows high-RUL values to be represented and materially helps FD002 and FD004. It changes average bias from underprediction to optimism on FD001 and FD003, so it is not described as universally better.',styles['BodyText']),Spacer(1,8),table([['Subset','Cap100 RMSE','Cap200 RMSE','Cap100 bias','Cap200 bias']]+[[s,fmt(cap100[cap100.subset==s].rmse.iloc[0]),fmt(cap200[cap200.subset==s].rmse.iloc[0]),fmt(cap100[cap100.subset==s].bias_predicted_minus_actual.iloc[0]),fmt(cap200[cap200.subset==s].bias_predicted_minus_actual.iloc[0])] for s in list(__import__('cmapss_data').SUBSETS)],[2.2*cm,3*cm,3*cm,3*cm,3*cm]),Spacer(1,8),Image(str(figdir/'final_cap200_actual_vs_predicted.png'),width=13*cm,height=10.8*cm),PageBreak()]
 specs=pd.read_csv(OUT/'specialist_official_metrics.csv');story += [Paragraph('Domain-aware experiments',styles['Heading1']),Paragraph('Specialists use a controlled subset identity for benchmarking only. This identity is not assumed to be available for a connected fleet. Each was selected between cap 100 and cap 200 using its own grouped validation.',styles['BodyText']),Spacer(1,7),table([['Subset','Specialist RMSE','Global cap200 RMSE','Specialist PHM08']]+[[r.subset,fmt(r.rmse),fmt(cap200[cap200.subset==r.subset].rmse.iloc[0]),fmt(r.phm08_mean)] for r in specs[specs.subset!='Pooled'].itertuples()],[3*cm,3.3*cm,3.8*cm,3.3*cm]),Spacer(1,9),Paragraph('Research implementation results',styles['Heading2']),Paragraph('The source-pretrain plus target-residual transfer baseline was tested for FD002/FD004 and rejected. The CPU-bounded compact TCN implemented three dilated residual blocks plus sensor and temporal attention; it was also rejected. The dual-scale Transformer was deliberately not run because the preceding simpler sequence model did not justify more tuning.',styles['BodyText']),Spacer(1,7),table([['Experiment','Subset','Validation RMSE','Validation PHM08']]+[[r.model,r.subset,fmt(r.rmse),fmt(r.phm08_mean)] for r in pd.concat([transfer,tcn]).itertuples()],[7*cm,2.2*cm,3.2*cm,3.2*cm]),PageBreak()]
 crit=pd.DataFrame(critical);story += [Paragraph('Critical range and operating workflow',styles['Heading1']),Paragraph('The 0-30 cycle range is reported separately because maintenance decisions become time critical there. The table uses raw official targets and predictions.',styles['BodyText']),Spacer(1,7),table([['Model','Subset','Cases','RMSE','MAE','PHM08','Within 10 cycles']]+[[r.model,r.subset,str(r.cases),fmt(r.rmse),fmt(r.mae),fmt(r.phm08_mean),fmt(r.within_10_cycles_fraction)] for r in crit.itertuples()],[3.2*cm,2*cm,1.3*cm,1.8*cm,1.8*cm,2*cm,2.3*cm]),Spacer(1,10),Paragraph('Competition pipeline',styles['Heading2']),Paragraph('Sensor telemetry -> training-fitted preprocessing -> operating-condition identification -> RUL prediction -> normal-reference reconstruction score -> deterministic triage -> idempotent mock work order -> MLflow. The run used '+str(meta['reference_windows'])+' early-life reference windows, scored '+str(meta['stream_windows'])+' stream windows and emitted '+str(meta['work_orders'])+' work orders.',styles['BodyText']),Spacer(1,7),Paragraph('USP',styles['Heading2']),Paragraph('The system combines a model that handles observable operating-condition changes without subset identity, causal feature controls and a complete explainable maintenance workflow. RUL prediction and anomaly reconstruction are separated, which makes the system honest about the absence of anomaly labels in C-MAPSS.',styles['BodyText']),Spacer(1,8),Paragraph('Recommendation',styles['Heading1']),Paragraph('Use the global cap-200 HGB as the benchmark candidate. Its better pooled result comes with documented FD001/FD003 tradeoffs. Keep the failed specialist, transfer, TCN and routing results in the evidence pack rather than presenting them as improvements. Before live use, calibrate uncertainty, set maintenance thresholds with domain experts and monitor operating-condition drift.',styles['BodyText'])]
 SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=1.25*cm,leftMargin=1.25*cm,topMargin=1.25*cm,bottomMargin=1.25*cm,title='Domain Aware C MAPSS Final Report').build(story)
 print(PDF)
if __name__=='__main__':main()
