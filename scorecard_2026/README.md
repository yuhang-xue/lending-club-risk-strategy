# 当前评分卡研究复现

原正式运行：20261111T144556。公开版适配数据入口和来源说明路径，不改变建模与策略逻辑。results/ 是正式运行聚合快照，不是本次重训结果。

## 环境与运行

Python 3.13；精确依赖见 requirements-original-lock.txt。LightGBM 需与平台匹配的 OpenMP（macOS 通常安装 libomp）。

```sh
cd scorecard_2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --stage audit --data /path/to/accepted_2007_to_2018Q4.csv.gz
python run.py --data /path/to/accepted_2007_to_2018Q4.csv.gz
```

每次创建 reports/<run_id> 和 artifacts/<run_id>，不覆盖已有运行。完整流程先测试，再审计、训练、冻结、评估并独立核验。

```sh
python verify.py --run reports/<run_id>
python supplement_business.py
```

补充脚本读取 latest_run.json，在基础运行后生成 FICO 策略、分组校准与评分案例；补充目录必须尚不存在。完整115项核验依赖本地重建的逐笔预测、模型及原始数据，不能仅靠公开聚合表重算。

original_verification.json 与 original_execution_checks.json 为历史记录；本次代码测试另存 publication_checks.json。本次未完整重训。

## 研究边界

见[来源说明](provenance.md)。研究36期已决贷款，统一快照未知，2015批次曾用于旧研究，仅保证本轮开发隔离。概率是终态事件概率，不是一年违约率。FICO比较、分组诊断及案例为事后描述扩展，无线上审批或利润结论。

旧版根目录 src/、results/ 等属于四字段研究，不与本目录混用。
