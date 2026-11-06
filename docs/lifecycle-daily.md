# 每日生命周期策略

固定调用 `lifecycle_5y_full_wm` v2.3.0：优先以全部可用历史的月K判断A/B/C/D/E/F/G/H，再与最近5年（月K）窗口交叉确认；月K不足时明确降级为周K，不再把1年窗口当作慢周期替代。一致才入围，冲突和不足历史单列。每次扫描和任务manifest记录同一策略契约，任务拒绝发布契约不一致的结果。

Windows任务名称：`AlphaSift-Lifecycle-Daily`。每天机器本地时间07:00运行；当前机器为Pacific Standard Time，随夏令时调整。使用当前用户交互式登录身份，无密码存储、不提升权限。需要电脑开机、用户登录及网络；启用错过计划后补跑，禁止任务重叠，最长18小时。不下单、不访问交易账户。

安装：在项目根目录运行 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install-lifecycle-daily.ps1`。已有同名任务时安装脚本拒绝覆盖。

手动执行：`.venv\Scripts\python.exe -m alphasift.lifecycle_daily`。
检查依赖及交易日（不抓行情）：同一命令加 `--check`。

执行顺序：重新获取CN/US股票快照 → 当前过滤范围内下载全历史日线 → 最近5年及全部可用历史分别判断A/B/C/D/E/F/G/H → 汇总所有一致候选 → 获取财务报表 → 每市场每行业综合前5。

指标为日K指标、周K确认转折和月K慢周期确认。A/H必须处于月线低位区，D必须处于月线高位且月均线向上，B/C/E/F/G必须分别满足相应的月线多头/空头反弹结构，其中C/F/G要求比B/E多确认一到两个已确认周线转折点，代表同一周期沿时间轴更深入的位置（详见 `docs/lifecycle-crosscheck.md`）。现有过滤：价格≥1、成交额≥2000万；A股去除ST/退市名称，美股市值≥10亿美元。不是所有上市股票无条件全覆盖。行业名称映射沿用已有文件，可能过时，新增代码无法匹配时保留未分类。

每次运行保存在 `data/daily-runs/<UTC时间>/`，包括 `manifest.json` 状态、`run.log` 日志及 `data/` 完整报告。任一市场下载失败比例超过20%、没有一致候选或财务结果为空时失败，不更新成功报告指针。所有失败数仍需查看当次报告，低于阈值也不等于完全覆盖。

成功后的固定入口：`data/lifecycle-daily-latest.html`；机器可读指针：`data/daily-runs/latest-success.json`。首次每日任务成功前，这两个文件可能尚未存在，原 `data/lifecycle-industry-top5.html` 历史报告不被覆盖。日期取自每市场当次实际收盘日；周末/节假日允许重复最近交易日。全历史仍仅指供应商全部可用历史。

检查任务：`Get-ScheduledTaskInfo -TaskName AlphaSift-Lifecycle-Daily`。
暂停：`Disable-ScheduledTask -TaskName AlphaSift-Lifecycle-Daily`。
恢复：`Enable-ScheduledTask -TaskName AlphaSift-Lifecycle-Daily`。

此次配置验证与单元测试不代表新一轮全市场抓取已完成；以后每次以运行manifest的success状态及实际覆盖数判断结果。
