# 数据来源与发布适配

来源：https://www.kaggle.com/datasets/wordsforthewise/lending-club
文件：accepted_2007_to_2018Q4.csv.gz，2,260,701 行、151 列。
SHA-256：55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d。
本地历史审计显示部分记录最晚至2019-03、上传日期2019-04-10，不能证明统一状态快照。

公开版通过 --data 指定原始文件，移除作者旧电脑路径依赖；仍强制核对原始文件哈希。publication_provenance 只记录本说明，不冒充历史审计证据。建模、分箱、标签与策略逻辑保持不变。未重新完整训练；发布代码哈希不是原运行冻结哈希。
不分发原始数据、逐笔预测、ID清单和模型；由使用者自行获取数据后本地重建。
