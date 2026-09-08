# Changelog

## Unreleased

- `lifecycle_5y_full_wm` 升级到 v2.2.0：补齐 C（第二次已确认高点回踩，一个周期高于B）、F（D后第二个更深低点及反弹，一个已确认Lower High）、G（D后第三个更深低点及反弹，两个已确认Lower High）三个阶段，完成 A/B/C/D/E/F/G/H 全阶段扫描，A股和美股均适用；`STAGES` 契约、日报和前端筛选一并更新
- 支持 DSA 通过 `context["dsa"]` 注入候选 provider，AlphaSift 会在 L1 初筛后、LLM 重排前补充 DSA 行情、基本面和新闻上下文
- `dsa_adapter.screen()` 现在会透传 DSA context，并在候选结果中保留 `dsa_context`、`dsa_news` 和 `dsa_analysis_summary`
- LLM ranking prompt 会读取候选上的 DSA provider context，便于排序阶段利用 DSA 已有数据能力

## 2026-04-12

- 明确说明 `DSA` 指外部项目 `daily_stock_analysis`，补充两者的职责边界与调用关系
- 更新 README、Skill 文档和设计说明，说明 DSA 只在最终入围候选上调用
- 修正文档中过期描述：移除对 `shrink_pullback` 可直接运行和“未实现 L3 deep_analysis”的错误说法
- 补充当前 DSA overlay 行为说明：结构化结果会在最后阶段影响 `final_score`、风险判断和最终排名
