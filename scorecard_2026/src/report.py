"""All report numerics and figures are generated from saved result tables."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def report(out,a,cfg):
 t=out/'tables';f=out/'figures';f.mkdir(exist_ok=True)
 read=lambda name:pd.read_csv(t/(name+'.csv'))
 fmt=lambda df:df.to_markdown(index=False,floatfmt='.4f')
 meanings={'loan_amnt':'贷款申请金额','annual_inc':'年收入','dti':'债务收入比','delinq_2yrs':'过去两年逾期次数','inq_last_6mths':'过去六个月征信查询次数','open_acc':'开放信用账户数','pub_rec':'负面公共记录数','pub_rec_bankruptcies':'公共破产记录数','tax_liens':'税收留置记录数','mort_acc':'抵押贷款账户数','total_acc':'信用账户总数','revol_util':'循环信用使用率','revol_bal':'循环信用余额','purpose':'贷款用途','home_ownership':'住房持有状态','addr_state':'居住州','emp_length':'就业年限','fico_mean':'原始FICO区间均值','log_income':'年收入log1p','loan_income':'贷款金额与年收入比','revol_income':'循环余额与年收入比','credit_history_months':'信用历史月数','income_invalid':'无效或缺失收入标记','fico_range_low':'原始FICO区间下界','fico_range_high':'原始FICO区间上界','earliest_cr_line':'最早信用账户月份','verification_status':'收入核验状态，时点不明而禁用'}
 transforms={'fico_mean':'(fico_range_low+fico_range_high)/2; bounds 300..850 and low<=high else missing','log_income':'log1p(annual_inc) only income>0','loan_income':'loan_amnt/annual_inc only income>0','revol_income':'revol_bal/annual_inc only income>0','credit_history_months':'12*(issue_year-earliest_year)+issue_month-earliest_month; negative -> missing','income_invalid':'1 if annual_inc missing or <=0 else 0','emp_length':'extract integer years; <1 year ->0; unknown->missing','revol_util':'strip percent sign; numeric conversion; negative->missing'}
 registry=read('feature_registry')
 registry['meaning']=registry.field.map(meanings).fillna('not admitted; retained for source audit only')
 registry['transformation']=registry.field.map(transforms).fillna('frozen training categories; missing/unknown explicit' if False else 'see role; numeric coercion and nonfinite->missing for admitted numeric fields')
 registry.loc[registry.field.isin(['purpose','home_ownership','addr_state']),'transformation']='string category; missing explicit; levels fit on training only'
 csv(t/'feature_registry.csv',registry)

 plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'savefig.bbox':'tight'})
 met=read('model_comparison');oot=met[(met.split=='OOT')&met.model.isin(['FICO','LR','LightGBM'])];pol=read('threshold_oot');ps=read('psi_summary');cal=read('calibration_report');ks=read('ks_curve_oot');curves=read('threshold_curve_oot');sh=read('shap_global').sort_values('mean_abs_shap').tail(15)
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for name in ['LR','LightGBM']:
  k=ks[ks.model==name];axs[0].plot(k.fpr,k.tpr,label=name);axs[1].plot(k.risk_threshold,k.ks,label=name)
 axs[0].plot([0,1],[0,1],'--',color='gray');axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='OOT ROC')
 axs[1].set(xlabel='PD threshold (descriptive only)',ylabel='TPR - FPR',title='OOT KS');axs[1].set_xlim(0,1)
 for ax in axs:ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(f/'roc_ks.png');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for name in ['LR','LightGBM']:
  c=cal[(cal.model==name)&(cal.split=='OOT')&(cal.n>0)];stable=c[c.n>=200];sparse=c[c.n<200]
  line=axs[0].plot(stable.mean_pd,stable.actual,'o-',label=name)[0]
  if len(sparse):
   axs[0].scatter(sparse.mean_pd,sparse.actual,facecolors='none',edgecolors=line.get_color(),marker='s',s=55,label=name+' sparse (<200)')
   for row in sparse.itertuples():axs[0].annotate('n='+str(row.n),(row.mean_pd,row.actual),xytext=(4,6),textcoords='offset points',fontsize=8)
  c=curves[curves.model==name];axs[1].plot(c.acceptance,c.actual_bad_rate,label=name)
  c=pol[pol.model==name];axs[1].scatter(c.acceptance,c.actual_bad_rate,s=45)
 axs[0].plot([0,.65],[0,.65],'--',color='gray');axs[0].set(xlabel='Mean predicted PD',ylabel='Observed bad rate',title='OOT calibration',xlim=(-.02,.65),ylim=(-.04,.65))
 axs[1].set(xlabel='Simulated acceptance',ylabel='Accepted observed bad rate',title='OOT descriptive curve; dots = frozen policies')
 for ax in axs:ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(f/'calibration_policy.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));ax.barh(sh.field,sh.mean_abs_shap,color='#34699a');ax.set(xlabel='Mean absolute TreeSHAP (raw log odds)',title='Frozen LightGBM | fixed OOT sample');fig.tight_layout();fig.savefig(f/'shap_global.png');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for j,name in enumerate(['LR','LightGBM']):
  q=ps[ps.model==name];axs[j].bar(q.reference+' -> '+q.current,q.psi,color='#34699a');axs[j].axhline(.1,color='orange',ls='--',label='investigation level 0.1');axs[j].set(title=name+' score PSI',ylabel='PSI');axs[j].legend()
 fig.tight_layout();fig.savefig(f/'psi.png');plt.close(fig)
 split=read('split_summary');selection=read('feature_selection');cond=read('conditional_tasks');maturity=read('maturity_residual_summary');main=maturity[maturity.year.astype(str)=='all'].iloc[0]
 best=oot.sort_values('auc',ascending=False).iloc[0];lr=oot[oot.model=='LR'].iloc[0];lg=oot[oot.model=='LightGBM'].iloc[0];fico=oot[oot.model=='FICO'].iloc[0]
 shift=read('population_shift').set_index(['split','field']);train_fico=shift.loc[('train','fico_mean'),'mean'];oot_fico=shift.loc[('OOT','fico_mean'),'mean'];train_dti=shift.loc[('train','dti'),'mean'];oot_dti=shift.loc[('OOT','dti'),'mean'];train_inc=shift.loc[('train','annual_inc'),'median'];oot_inc=shift.loc[('OOT','annual_inc'),'median']
 lines=f'''# LendingClub 风险评分卡完整研究报告

**研究性质：已结清/已确认坏样本的放款批次回溯研究。统一快照未证实；不是严格历史回测，不是全部申请人审批验证。**

生成时间：{now()}。运行编号：`{out.name}`。本报告数字全部从本次结果表生成，原执行方案保持不变。

## 1. 研究结果与结论

本次在36期、假设账龄筛查后的已决贷款上完成WOE＋LR、FICO基准和LightGBM流程对照。较晚放款批次留出集AUC最高为 {best['model']}（{best['auc']:.4f}）。LR相对FICO的AUC差为 {lr['auc']-fico['auc']:+.4f}，LightGBM相对LR差为 {lg['auc']-lr['auc']:+.4f}。这是同信息范围下的完整流程比较，不能将全部差异归因于算法。

{fmt(oot[['model','n','bad_rate','auc','ks','brier','ece','mean_pd']])}

留出集实际坏样本率 {lr['bad_rate']:.2%}，LR平均预测 {lr['mean_pd']:.2%}，LightGBM平均预测 {lg['mean_pd']:.2%}，分别低估 {(lr['bad_rate']-lr['mean_pd'])*100:.2f} 和 {(lg['bad_rate']-lg['mean_pd'])*100:.2f} 个百分点。因此排序改善不代表概率已准确，不建议把当前PD直接当作未来风险承诺。FICO只是排序基准，不报告伪PD或Brier。

## 2. 数据与时间证据

数据源为Kaggle wordsforthewise/lending-club本地 accepted_2007_to_2018Q4.csv.gz，2,260,701行、151列。SHA-256与此前来源冻结一致，完整证据见[data_audit.md](data_audit.md)及source_metadata.json。

既有来源证据只确认记录最晚2019-03及2019-04上传，不能证明统一状态截至日期。2019-03是**假设参考月**，6个月为研究缓冲假设，不是已验证的充分成熟期。故按方案第四节降级为已结清/已确认坏样本研究。历史快照无法证明本次训练标签在模拟上线时已可得；last_pymnt_d不作为标签确认日。

旧研究已经使用2015批次。本次OOT只保证不进入本轮训练、调参、校准和策略选择，不能称项目层面的首次未见数据或独立外部验证。

## 3. 样本、成熟度与切分

{fmt(split)}

Fully Paid=0；Charged Off/Default=1。主模型不纳入Current、Grace、Late和信用政策不符系列。异常行、期限、账龄假设、状态及时间范围按固定优先级生成互斥流水账，见sample_flow.csv和sample_ledger.csv.gz。

假设主账龄范围、常规政策状态的完整同批次共 {int(main.n):,} 笔，残余未决 {int(main.unresolved)} 笔，占 {main.unresolved_share:.6%}；已决坏样本率 {main.rate_resolved:.4%}。未决全部好/全部坏下的批次坏样本率范围为 {main.bad_lower:.4%}—{main.bad_upper:.4%}。范围窄不能证明不存在未知快照或选择偏差。3/6/9个月结果见maturity_sensitivity.csv；已决/未决收入、FICO、DTI及贷款金额分布见resolved_unresolved_comparison.csv。

member_id全部缺失，检查已执行但无法隔离借款人，只保证loan ID跨集不重复。V_model/V_cal/V_policy均在训练前检查至少2000笔、好坏各200笔支持量；这只是研究门槛，不是精度保证。

## 4. 贷前特征与WOE评分卡

仅显式白名单和经审核衍生字段进入模型；平台评级、定价、月供、贷后偿还/结算、ID、标签均不进入模型。verification_status额外排除：无法证实初始申请时已知。FICO使用原始区间平均，不使用last_fico。信用历史以issue_d近似申请月；缺少逐笔字段更新史，不能证明所有字段完全重现历史输入。

收入≤0及非法数值不进入比率计算；无穷转缺失；负信用历史转缺失并记录异常。LR缺失独立箱、未见值WOE=0；树模型保留缺失，未见类别转缺失。类别水平仅在训练拟合。

训练集最多20个分位初箱；相邻合并使用平滑IV损失最小及固定左侧决胜；普通箱≥5%、最多5箱；连续变量训练坏账率单调。缺失箱例外披露。WOE=ln(好分布/坏分布)，每个有观测箱好坏各加0.5。IV≥0.02及训练WOE相关绝对值<0.7筛选。

最终LR保留 {int(selection.selected.sum())} 个字段：{', '.join(selection.loc[selection.selected,'field'])}。

{fmt(selection[['field','iv','ordinary_bins','selected','reason']])}

分箱合并记录、边界、系数、缺失例外和跨期单调性分别见binning_merge_log.csv、binning_table.csv、lr_coefficients.csv、bin_monotonicity.csv。系数反向需结合相关性和联合效应解释，不能更改数据迎合预期。

## 5. 模型开发、校准与评分

LR在四个C值中以V_model AUC优先、logloss次优、较小C决胜；LightGBM仅三套预设结构参数，最多600轮，V_model早停40轮。保留各自合理的特征流程；未开展同变量控制实验，不主张纯算法增益。训练耗时和树数见runtime_cost.csv。

校准触发：V_model十个等宽概率箱的ECE>0.01；只在独立V_cal拟合预设正斜率Platt，不在V_policy/OOT选择校准。有限样本的ECE依赖分箱，不是概率已准确的保证。

{fmt(read('calibration_decisions'))}

校准前后各集合指标均保存在model_comparison.csv。正斜率保证排序不变；若跨期校准仍偏移，应报告而非用OOT重校准。

Score=600−(20/ln2)×ln50+(20/ln2)×ln((1−PD)/PD)。600分对应1/51≈1.96%、580分≈3.85%、620分≈0.99%，是理论转换而非实测风险。PD只裁剪到[1e-9,1−1e-9]以数值稳定；策略分数取两位小数，同分整体处理。

本次两个模型均未触发校准，V_cal按预设规则闲置，没有挪作其他开发用途。实现支持正斜率Platt作用于logit的仿射变换，但本次加分表来自未校准LR：各变量points相加，再加一次base_points；scorecard_points.csv记录其定义。理论可加性在未触发极端裁剪且取整前成立。PD对应本研究终态标签，不等于一年PD。

## 6. OOT、稳定性与策略

![ROC与KS](figures/roc_ks.png)

KS最大值阈值只作描述统计，不用于策略选择。校准图中不足200条的概率箱仅用空心方块展示且标注样本量，不与主曲线相连；完整箱计数仍保留在calibration_report.csv。这是展示稳定性标记，未改变冻结的ECE计算或校准触发规则。季度指标见quarterly_metrics.csv；样本少的分组标记不稳定，本轮未声称统计显著增益。

![概率与策略](figures/calibration_policy.png)

阈值只在V_policy按50/70/90%演示目标选择，通过率不超过目标且最接近目标；同分不拆分。最低通过数max(500,5%×V_policy样本量)。分数阈值冻结后原样应用到OOT：

{fmt(pol[['model','target','threshold','accepted','acceptance','actual_bad_rate','mean_pd','good_reject','bad_capture','bad_escape']])}

OOT通过率可偏离验证目标。坏账率为通过坏样本/通过样本；好客户误拒分母为全部好样本；坏客户拦截/漏放分母为全部坏样本，两者合计1。描述性Wilson区间见threshold_oot.csv；它未修正多阈值选择，也未处理不可识别的借款人依赖，不保证未来风险。无外部坏账目标，本次不宣称风险目标可控或业务最优。没有利润、LGD、资金成本，不能宣称收益或损失金额。

完整OOT阈值曲线仅诊断，不能据此重新挑选最佳策略并当独立验证。

{fmt(ps)}

![分数PSI](figures/psi.png)

训练到OOT，平均FICO由 {train_fico:.2f} 变为 {oot_fico:.2f}，平均DTI由 {train_dti:.2f} 变为 {oot_dti:.2f}，收入中位数由 {train_inc:,.0f} 变为 {oot_inc:,.0f}。FICO下降及DTI上升与风险上移方向一致，但不是因果证明。三项主样本缺失率均为0，期限固定36期；其他字段仍需结合跨年覆盖检查。全部分数PSI低于0.1，但留出概率明显低估，说明分数分布稳定不等于概率稳定。

PSI使用训练分数固定边界和0.5平滑；0.1是调查线，不是失效判定。发生漂移需结合feature_missingness_by_year.csv、population_shift.csv中的FICO、收入、DTI、期限与缺失结构，连同区分度和校准变化判断。验证集合并用于PSI描述，不改变各开发子集职责。

## 7. 冻结模型解释

![SHAP摘要](figures/shap_global.png)

使用LightGBM原生TreeSHAP、训练树路径计数作为背景，固定随机种子抽取最多2000条OOT样本，不使用OOT构造背景或反向筛特征。SHAP贡献处于**未校准基础模型的log-odds尺度**，贡献与基准之和已与原始输出核对。

shap_cases.csv包含低、中、高风险与70%策略下的好客户误拒、坏客户漏放案例，固定选取规则；这些是模型归因和理由码演示，不是因果关系或真实业务拒绝建议。

## 8. 条件项与验收边界

{fmt(cond)}

关键代码测试见test_results.txt；结果独立重算见verification.json。执行检查覆盖样本对账、ID隔离、字段白名单、分箱约束、冻结顺序、重载一致性、分数反算及策略同分处理。原始数据未改写，原执行方案未改写。

**实施主线完成不等于数据限制消失。** M1完成来源和可研究性审计，但无法通过“已确认成熟快照”的严格标准；M2—M6按降级回溯口径执行。不可完成的强验收项（严格历史回测、确认成熟期、借款人隔离、全部申请人泛化）明确不通过，不因流程跑通而冒称通过。

## 9. 复现入口

项目根目录README说明环境、入口及单独核验命令。artifacts/{out.name}保存配置、模型、源码快照、来源哈希、依赖版本、随机种子及模型/策略冻结记录。模型仅加载本项目可信joblib文件。

表格数字由代码生成，完整报告与面试摘要共用同一结果表。原始数据仅引用既有本地文件，不纳入公开分发。
'''
 md(out/'完整研究报告.md',lines)
 md(out/'面试展示摘要.md',f'''# LendingClub 评分卡项目｜面试摘要

**项目定位：已放款、36期已决贷款的批次回溯研究，不是生产审批模型或严格历史回测。**

完成原始数据审计、标签与排除流水账、分区开发、训练集WOE单调分箱/IV筛选、LR评分卡、LightGBM和FICO基准、校准检验及阈值模拟。开发集未触发预设校准条件，未拟合校准器。

训练截至2013年，2014年按调参/校准/策略拆分，2015年1—9月留出检验。留出样本 {int(lr.n):,}，坏样本率 {lr.bad_rate:.2%}。

{fmt(oot[['model','auc','ks','brier','ece']])}

LR筛选 {int(selection.selected.sum())} 个变量；按600分、好坏赔率50、PDO20构建分数。阈值在策略验证集冻结，以50/70/90%模拟通过率为演示目标，再原样应用留出集，避免在测试集重取分位点。

{fmt(pol[pol.target==.7][['model','threshold','acceptance','actual_bad_rate','good_reject','bad_capture']])}

留出集LR/LightGBM平均预测分别为 {lr.mean_pd:.2%}/{lg.mean_pd:.2%}，均低于实际 {lr.bad_rate:.2%}；低PSI未阻止概率低估，不能将当前概率视为未来风险承诺。

可展示证据：训练分箱及系数、模型比较、校准图、PSI、冻结阈值效果、TreeSHAP案例和一条命令复现。

必须说明：统一快照未确认，假设账龄筛查不能证明充分成熟；member_id全缺失；只研究平台已放款且已决样本，存在选择偏差。2015批次被旧项目使用过，本轮只是流水线隔离留出。PD不是一年PD；未计算利润或金额损失；所有差异属于流程对照，未宣称纯算法增益或显著性。
''')
 md(out/'strategy_recommendations.md',f'''# 策略建议与限制

当前只支持研究演示。留出集实际坏样本率 {lr.bad_rate:.2%}，LR/LightGBM平均预测仅 {lr.mean_pd:.2%}/{lg.mean_pd:.2%}；两模型均存在低估。分数PSI小并不能替代概率检验。对50/70/90%档位同时展示通过率、通过客群坏账率、平均PD、好客户误拒与坏客户拦截，不能只选一个好看的指标。以threshold_validation.csv为阈值来源，以threshold_oot.csv为留出表现。

没有外部业务风险目标，因此不指定“最优”档位，不依据OOT曲线重新挑选阈值。若未来获得真实目标，应使用新的独立开发/检验安排，并补齐申请总体、拒绝样本、损失/收入成本及快照证据后再讨论上线。对模型漂移首先核查人群和字段覆盖，再判断是否需要重训或重新校准；不得在本次OOT修正后仍称独立验证。
''')
