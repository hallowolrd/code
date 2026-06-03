# AGENTS.md

- 本项目用于 FL + MoE 实验。
- 第一版只复刻 `moefedavg.py` 中仍在使用的核心 baseline。
- 凡是原文件注释中明确写了“已删除 / 关闭 / 不再使用 / 移除”的功能都不要重新实现。
- 后续聚合会拆成 `non_expert_agg_method` 和 `expert_agg_method` 两套接口。
- 默认 baseline 是：
  - `non_expert_agg_method: uniform`
  - `expert_agg_method: uniform`
