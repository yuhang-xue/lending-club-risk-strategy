# 四字段WOE评分卡：公开实现

本模块提供binning、WOE/IV、四字段逻辑回归、PDO分数映射、风险排序、等量筛选、配对Bootstrap和固定箱PSI。代码来自2026-11-02的本地评分卡实现，保留原核心算法；公开运行适配器为`src/run_scorecard.py`。

## 查看已有证据

- [评分卡方法与结果](../docs/评分卡方法与结果.md)
- [聚合结果目录](../results/scorecard/)：36行情景、6项区间、固定卡PSI和逐箱统计；没有逐笔贷款或模型。
- [导出清单](../results/scorecard/publication_manifest.json)：逐文件字节数、哈希与源实验manifest哈希。导出数据逐字节核对历史manifest，未修改封存实验。

## 运行

在仓库根目录使用`requirements-scorecard.txt`创建独立环境，先运行`src/run.py`生成基线，再运行：

```sh
.venv-scorecard/bin/python src/run_scorecard.py --data data/raw/LC_loans_granting_model_dataset.csv --baseline-run outputs/baseline_for_scorecard --output outputs/scorecard_reproduction
```

完整命令见[首页](../README.md)。适配器先执行基线audit/verify.py，核对原始CSV哈希与贷款身份，再拟合三张评分卡并重放6,000次Bootstrap，核对历史整数、区间与固定卡PSI。输出不存在才创建；验证失败保留失败记录，不能当作通过。

```sh
.venv-scorecard/bin/python -m unittest discover -s scorecard_v1/tests -v
.venv-scorecard/bin/python audit/verify_scorecard_public.py
```

12项合成边界测试覆盖训练外类别、缺失与保留类别、分箱边界、训练/评价隔离、WOE公式、可加分值、PDO、同分排序、重采样边界与PSI。

本模块不公开原本依赖私有历史manifest路径的run/audit/data/delivery脚本，也不要求使用者伪造这些输入。新适配器以公开CSV重建输入，历史核心算法与导出的历史结果保持可追溯。

80%是终态贷款研究样本保留率，评分并非校准后绝对风险。PSI描述分布差异，不证明模型失效；所有评价年份已查看，不构成独立验证。
