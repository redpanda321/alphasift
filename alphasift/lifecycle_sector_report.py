"""Group ALL cross-window agreement candidates by market/sector/stage."""
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path

STAGES = "ABDEH"
LABELS = {"A": "底部候选", "B": "首次上涨回调", "D": "周期高位候选",
          "E": "顶部后首次反弹", "H": "长期下降末端"}
US_SECTORS = {"Technology": "科技", "Communication Services": "通信服务",
              "Industrials": "工业", "Consumer Cyclical": "可选消费",
              "Consumer Defensive": "必需消费", "Healthcare": "医疗保健",
              "Energy": "能源", "Financial Services": "金融服务",
              "Basic Materials": "基础材料", "Real Estate": "房地产", "Utilities": "公用事业"}


def build(base=Path("data")):
    supplement = {r["code"]: r for r in json.loads((base / "sector-cn-supplement.json").read_text(encoding="utf-8"))}
    groups = defaultdict(list)
    coverage = []
    all_candidates = []
    sectors_by_market = {}
    for market in ("cn", "us"):
        raw = json.loads((base / f"lifecycle-crosscheck-{market}.json").read_text(encoding="utf-8"))
        prior = json.loads((base / f"sector-scan-{market}.json").read_text(encoding="utf-8"))
        metadata = {r["code"]: r for r in prior["rows"]}
        with (base / f"sector-snapshot-{market}.csv").open(encoding="utf-8-sig", newline="") as stream:
            names = {r["code"]: r["name"] for r in csv.DictReader(stream)}
        sectors_by_market[market] = set(prior["sectors"])
        states = Counter(r["status"] for r in raw["rows"])
        assert sum(states.values()) == raw["attempted"] == len({r["symbol"] for r in raw["rows"]})
        coverage.append(dict(market=market, as_of=raw["as_of"], snapshot=raw["snapshot_count"],
                             attempted=raw["attempted"], **states))
        for r in raw["rows"]:
            if r["status"] != "AGREEMENT":
                continue
            symbol = r["symbol"]
            code = symbol.rsplit(".", 1)[0] if market == "cn" else symbol
            sectors = [s for s in metadata.get(code, {}).get("sectors", []) if s != "未分类"]
            source = "新浪行业" if market == "cn" else "Yahoo sector"
            if not sectors and market == "cn" and supplement.get(code, {}).get("industry"):
                sectors = ["补充行业：" + supplement[code]["industry"]]
                source = "Yahoo公司行业补充"
            if not sectors:
                sectors, source = ["未分类"], "无已保存分类"
            sectors = sorted(set(sectors))
            for sector in sectors:
                row = dict(market=market, sector=sector, sector_source=source,
                           stage=r["consensus_stage"], code=code, name=names.get(code, symbol),
                           price=r["five_year"]["price"], currency="CNY" if market == "cn" else "USD",
                           as_of=r["as_of"], five_score=r["five_year"]["scores"][r["consensus_stage"]],
                           full_score=r["full_history"]["scores"][r["consensus_stage"]],
                           score=r["consensus_score"], full_start=r["full_history"]["history_start"])
                groups[(market, sector, row["stage"])].append(row)
                all_candidates.append(row)
                sectors_by_market[market].add(sector)
    ranked = []
    summary = []
    date_label = '；'.join(f"{r['market']} {r['as_of']}" for r in coverage)
    lines = ["# A股 / 美股 · 行业板块 · 阶段前10", "",
             f"数据截至{date_label}；全部双窗口一致候选作为排名池。每个市场×行业×阶段独立取前10，不足10只全部展示。",
             "分数为5年/全部可用历史两者较小值，不是胜率。日K复权收盘价，非实时价格。D是高位候选，不是已确认顶部。",
             "行业沿用已保存的新浪/Yahoo分类，缺失时使用已保存公司行业补充；未分类不强行推断。分类口径并不统一。", ""]
    sections = []
    for market in ("cn", "us"):
        market_name = "A股" if market == "cn" else "美股"
        lines += [f"## {market_name}", ""]
        for sector in sorted(sectors_by_market[market]):
            title = US_SECTORS.get(sector, sector)
            counts = {s: len(groups[(market, sector, s)]) for s in STAGES}
            summary.append(dict(market=market, sector=title, **counts))
            lines += [f"### {title}", ""]
            tables = []
            for stage in STAGES:
                pool = sorted(groups[(market, sector, stage)], key=lambda r: (-r["score"], r["code"]))
                chosen = [dict(r, rank=i+1) for i, r in enumerate(pool[:10])]
                ranked.extend(chosen)
                lines += [f"#### {stage} {LABELS[stage]}：共{len(pool)}只，展示{len(chosen)}只", ""]
                if not chosen:
                    lines += ["无符合候选。", ""]
                    continue
                headers = ["排名", "代码", "名称", "复权收盘价", "币种", "交易日", "5年分", "全历史分", "交叉分"]
                records = [[r[k] for k in ("rank", "code", "name", "price", "currency", "as_of", "five_score", "full_score", "score")] for r in chosen]
                lines += ["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)]
                for record in records:
                    lines.append("| " + " | ".join(str(v).replace("|", "/") for v in record) + " |")
                lines.append("")
                table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
                for record in records:
                    table += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in record) + "</tr>"
                tables.append(f"<h3>{stage} · {LABELS[stage]} <small>{len(pool)}只，展示{len(chosen)}只</small></h3>" + table + "</tbody></table>")
            count_text = " / ".join(f"{s} {counts[s]}" for s in STAGES)
            body = "".join(tables) or "<p>已保存分类中，本次没有双窗口一致候选；不表示行情覆盖完整。</p>"
            sections.append(f'<details class="sector" data-market="{market}"><summary>{market_name} · {html.escape(title)} <small>{count_text}</small></summary>{body}</details>')
    prefix = base / "lifecycle-sector-top10"
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    for suffix, rows in [(".csv", ranked), ("-all.csv", all_candidates)]:
        path = Path(str(prefix)+suffix)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>行业阶段前10</title>
<style>body{font:15px system-ui;margin:28px;color:#182536;background:#f7f9fc}header{position:sticky;top:0;background:#f7f9fc;padding:12px 0}input,select,button{padding:9px;margin:4px;border:1px solid #bbc8d8;border-radius:5px}details{background:white;border:1px solid #dce3ec;padding:14px;margin:12px 0;border-radius:8px;overflow:auto}summary{cursor:pointer;font-size:18px;font-weight:600}small{font-weight:400;color:#56667a;margin-left:12px}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{padding:8px;text-align:left;border-bottom:1px solid #edf0f5}th{background:#eff4fa}h3{margin-top:24px}</style>
<h1>A股 / 美股 · 行业板块 · 阶段前10</h1>
<p>截至 DATA_DATES。每个市场×行业×阶段独立取前10；复权日K收盘价，非实时行情。分数不是胜率，D仅为高位候选。</p>
<p>行业采用已保存新浪/Yahoo分类及公司行业补充，口径并不统一；未分类保留。下方无候选不等于该板块完整获取行情。</p>
<header><select id="market"><option value="">全部市场</option><option value="cn">A股</option><option value="us">美股</option></select><input id="query" placeholder="搜索行业、代码或名称"><button id="expand">展开可见板块</button><button id="collapse">收起全部</button></header>
'''+"".join(sections)+'''
<script>const blocks=[...document.querySelectorAll('.sector')];function filter(){let m=document.querySelector('#market').value,q=document.querySelector('#query').value.toLowerCase();blocks.forEach(b=>{b.hidden=!!((m&&b.dataset.market!==m)||(q&&!b.textContent.toLowerCase().includes(q)));if(q&&!b.hidden)b.open=true;});}document.querySelector('#market').onchange=filter;document.querySelector('#query').oninput=filter;document.querySelector('#expand').onclick=()=>blocks.forEach(b=>{if(!b.hidden)b.open=true});document.querySelector('#collapse').onclick=()=>blocks.forEach(b=>b.open=false);</script></html>'''
    prefix.with_suffix(".html").write_text(document.replace('DATA_DATES', html.escape(date_label)), encoding="utf-8")
    payload = dict(coverage=coverage, sectors=summary, selected_rows=len(ranked),
                   all_rows=len(all_candidates), ranked=ranked)
    prefix.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"coverage":coverage,"selected_rows":len(ranked),"all_rows":len(all_candidates),"sectors":summary},ensure_ascii=False))


if __name__ == "__main__":
    build()
