"""Render saved cross-window results without fetching or modifying scores."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [(Path(p), json.loads(Path(p).read_text(encoding="utf-8"))) for p in args.inputs]
    lines = ["# A/B/D/E/H 五年与全部历史交叉验证", "",
             "同一份新获取的复权日K分别按最近5个日历年、全部可用历史计算。",
             "一致性分取两个窗口对应阶段评分的较小值，不代表收益概率。D仅为高位候选。", "",
             "| 市场文件 | 截至交易日 | 快照数 | 请求历史数 | 一致 | 冲突 | 无匹配 | 历史不足以交叉 | 失败 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    all_rows = []
    for path, report in reports:
        counts = Counter(row["status"] for row in report["rows"])
        cells = [path.stem, report["as_of"], report["snapshot_count"], report["attempted"]]
        cells.extend(counts[k] for k in ["AGREEMENT", "CONFLICT", "NO_MATCH", "INSUFFICIENT_DISTINCT_HISTORY", "FAILED"])
        lines.append("| " + " | ".join(map(str, cells)) + " |")
        all_rows.extend(report["rows"])
    for stage in "ABDEH":
        matches = sorted([r for r in all_rows if r.get("consensus_stage") == stage],
                         key=lambda r: (-r["consensus_score"], r["symbol"]))
        lines += ["", f"## {stage}：一致候选 {len(matches)} 只（展示前20）", "",
                  "| 股票 | 复权收盘价 | 日期 | 五年分 | 全历史分 | 交叉分 | 全历史起点 |",
                  "|---|---:|---|---:|---:|---:|---|"]
        for r in matches[:20]:
            five, full = r["five_year"], r["full_history"]
            cells = [r["symbol"], five["price"], r["as_of"], five["scores"][stage],
                     full["scores"][stage], r["consensus_score"], full["history_start"]]
            lines.append("| " + " | ".join(map(str, cells)) + " |")
    conflicts = [r for r in all_rows if r["status"] == "CONFLICT"]
    lines += ["", "## 窗口分歧示例（前20）", "",
              "| 股票 | 五年阶段 | 全历史阶段 | 五年D日期 | 全历史D日期 |",
              "|---|---|---|---|---|"]
    for r in conflicts[:20]:
        five, full = r["five_year"], r["full_history"]
        lines.append(f"| {r['symbol']} | {five['stage']} | {full['stage']} | "
                     f"{five['evidence']['D_date']} | {full['evidence']['D_date']} |")
    lines += ["", "所有成功/失败及双窗口证据见同目录原始JSON。",
              "未验证逐交易日完整性、上市日起点和基本面。历史不足股票未计为交叉通过。",
              "股票范围受价格、成交额及美股市值过滤，不等于全部上市股票。", ""]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
