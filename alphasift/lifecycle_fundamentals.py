"""Fundamental overlay for the complete cross-window lifecycle candidate pool.

Run: uv run python -m alphasift.lifecycle_fundamentals
Legacy data scripts are parsed for their pure functions only, never executed.
"""
import ast
import csv
import html
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from alphasift.lifecycle_sector_report import US_SECTORS


def legacy_functions(path, names, namespace):
    """Load explicitly named function definitions without top-level file writes."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("Legacy function contract changed")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def score_candidate(stock, financial, rank_function):
    result, reason = rank_function(financial)
    row = dict(stock, financial=financial, eligible=False, exclusion=reason)
    if financial.get("financial_company"):
        row["exclusion"] = "金融企业：FCF/净债务口径不适用，缺少资本充足率/偿付能力等专门数据，待核验"
    if result:
        row.update(eligible=True, exclusion=None, fundamental_score=result["fundamental_score"],
                   combined_score=round(.8 * result["fundamental_score"] + .2 * float(stock["score"]), 2),
                   components={k: result[k] for k in ("fcf_component", "debt_component", "profit_component", "penalties")})
    return row


def main(base=Path("data"), report_only=False):
    import logging
    import threading
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    now = pd.Timestamp.now(tz="UTC")
    stocks = list(csv.DictReader((base / "lifecycle-sector-top10-all.csv").open(encoding="utf-8-sig")))
    assert len(stocks) == len({(s['market'], s['code']) for s in stocks})
    fetch = legacy_functions(base / "fundamental_scan.py", ("value", "dates", "analyze"),
                             dict(pd=pd, yf=yf, STOP=threading.Event(), cutoff=now.tz_localize(None).normalize()))["analyze"]
    ranker = legacy_functions(base / "rank_fundamental_top50.py", ("clip", "rank"),
                              dict(pd=pd, asof=now.tz_localize(None), translation=US_SECTORS))["rank"]
    # Same-day provider cache is explicitly retained, with original retrieval timestamps.
    cache = {}
    for filename in ("fundamentals-candidates.json", "lifecycle-fundamentals-all.json"):
        path = base / filename
        if path.exists():
            for f in json.loads(path.read_text(encoding="utf-8"))["rows"]:
                if report_only or (f.get("retrieved_at", "")[:10] == now.date().isoformat() and f.get("status") == "ok"):
                    cache[(f['market'], f['code'])] = f
    rows, requests = [], []
    for s in stocks:
        key = (s['market'], s['code'])
        if key in cache:
            rows.append(dict(cache[key], cache_reused=True))
        else:
            if report_only:
                raise ValueError(f"Report-only cache missing {key}")
            requests.append(dict(s, sectors=s['sector'], a_score=s['score'], h_score=s['score']))
    print(f"Candidate pool {len(stocks)}; same-day cached {len(rows)}; fetch {len(requests)}", flush=True)
    with (base / "lifecycle-fundamentals-fetch.jsonl").open("w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=2) as executor:
            for i, future in enumerate(as_completed([executor.submit(fetch, s) for s in requests]), 1):
                f = future.result()
                f.setdefault("retrieved_at", pd.Timestamp.now(tz="UTC").isoformat())
                rows.append(f)
                stream.write(json.dumps(f, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if i % 20 == 0:
                    print(f"Financial fetch {i}/{len(requests)}", flush=True)
    (base / "lifecycle-fundamentals-all.json").write_text(json.dumps(dict(started_at=now.isoformat(),
        finished_at=pd.Timestamp.now(tz="UTC").isoformat(), rows=rows), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lookup = {(r['market'], r['code']): r for r in rows}
    scored = [score_candidate(s, lookup[(s['market'], s['code'])], ranker) for s in stocks]
    groups = defaultdict(list)
    for r in scored:
        groups[(r['market'], r['sector'])].append(r)
    chosen, summaries = [], []
    date_label = '；'.join(sorted({s['market']+' '+s['as_of'] for s in stocks}))
    note = (f"以现有全部双窗口一致候选为范围，不是全市场基本面排名。行情截至{date_label}，日K复权收盘价，非实时。"
            "同日财务缓存保留原获取时间；Yahoo/yfinance年度报表，不是TTM，未逐份核对交易所公告，不能保证最新已披露。"
            "综合分=基本面80%+技术形态20%；基本面由FCF40分、负债30分、盈利30分及风险扣分组成。"
            "每市场×行业跨阶段取5，不足不补；D/E高分不是买入信号。金融企业专门指标缺失时暂不排名。"
            "行业沿用此前分类，粗细不统一，未分类单列而不冒称行业。金额单位为财报币种的亿，净债务负数表示净现金。"
            "本模型未包含估值、最新季度趋势、受限现金、到期债务与公告风险，入选不代表基本面未恶化或适合买入。")
    lines = ["# 各行业基本面综合前5", "", note, "", "## 评分细则", "",
             "门槛：最近年度FCF、经营现金流、净利润、收入、净资产均为正；三年连续年度记录及关键字段齐全；年报≤550天，资产负债表≤200天。",
             "FCF40：FCF率15（20%封顶）＋经营现金流/净利10（1倍封顶）＋三年正FCF占比10＋FCF同比5（-20%至20%线性）。",
             "负债30：净债务/年度FCF由0至5倍线性从30降至0。净利润30：净利率10（20%封顶）＋同比10（-20%至20%线性）＋三年正净利占比10。",
             "净利下降>20%、FCF下降>20%、已知股权激励/收入>10%，各扣5分；缺失股权激励不等于没有。所有比例按同一财报币种计算。", ""]
    sections = []
    headers = ["排名", "代码", "公司", "阶段", "综合", "基本面", "技术", "FCF(亿)", "总债务(亿)", "净债务(亿)", "净利率%", "财报币种", "年度截至", "负债截至", "财务获取UTC", "行情日"]
    for (market, sector), pool in sorted(groups.items()):
        eligible = sorted([r for r in pool if r['eligible']], key=lambda r: (-r['combined_score'], r['code']))
        top = [dict(r, rank=i+1) for i, r in enumerate(eligible[:5])]
        chosen.extend(top)
        summaries.append(dict(market=market, sector=sector, candidates=len(pool), eligible=len(eligible), selected=len(top)))
        title = f"{'A股' if market == 'cn' else '美股'} / {US_SECTORS.get(sector, sector)} — {len(top)}只 / 候选{len(pool)}只"
        lines += [f"## {title}", ""]
        table = []
        for r in top:
            f = r['financial']
            table.append([r['rank'], r['code'], r['name'], r['stage'], r['combined_score'], r['fundamental_score'], r['score'],
                          round(f['free_cash_flow']/1e8, 2), round(f['total_debt']/1e8, 2), round(f['net_debt']/1e8, 2),
                          round(f['net_margin_pct'], 2), f['currency'], f['period'], f['balance_period'], f['retrieved_at'], r['as_of']])
        if table:
            lines += ["|"+"|".join(headers)+"|", "|"+"---|"*len(headers)]
            lines += ["|"+"|".join(str(v).replace('|','/') for v in rec)+"|" for rec in table]
        else:
            lines += ["无通过本轮财务门槛的候选；不凑数。"]
        reasons = dict(Counter(r['exclusion'] for r in pool if not r['eligible']))
        lines += ["", f"未入围原因：{reasons}", ""]
        rendered = "<table><tr>" + ''.join(f"<th>{h}</th>" for h in headers) + "</tr>"
        rendered += ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in rec)+'</tr>' for rec in table)+"</table>"
        sections.append(f"<details open><summary>{html.escape(title)}</summary>{rendered}<p>{html.escape(str(reasons))}</p></details>")
    payload = dict(generated_at=pd.Timestamp.now(tz='UTC').isoformat(), scope=len(scored), selected=len(chosen),
                   summary=summaries, top5=chosen, all_rows=scored)
    stem = base / "lifecycle-industry-top5"
    audit_fields = ['market', 'sector', 'code', 'name', 'stage', 'eligible', 'exclusion',
                    'fundamental_score', 'combined_score', 'period', 'balance_period', 'retrieved_at',
                    'financial_currency', 'free_cash_flow', 'total_debt', 'cash', 'net_margin_pct', 'source']
    with (base / 'lifecycle-industry-top5-audit.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=audit_fields, extrasaction='ignore')
        writer.writeheader()
        for r in scored:
            f = r['financial']
            writer.writerow({**r, **{k:f.get(k) for k in audit_fields if k not in r}, 'financial_currency':f.get('currency')})
    stem.with_suffix('.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    stem.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    stem.with_suffix('.html').write_text('<!doctype html><meta charset="utf-8"><title>各行业综合前5</title><style>body{font:14px system-ui;padding:24px}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;border:1px solid #ddd}details{overflow:auto;margin:20px 0}summary{font-size:18px;font-weight:bold}p{line-height:1.8}</style><h1>各行业基本面综合前5</h1><p>'+html.escape(note)+'</p>'+''.join(sections), encoding='utf-8')
    flat = []
    for r in chosen:
        f = r['financial']
        flat.append({**{k:v for k,v in r.items() if k not in ('financial','components')}, **{'financial_'+k:v for k,v in f.items() if not isinstance(v,(dict,list))}})
    if flat:
        with stem.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in flat))))
            writer.writeheader(); writer.writerows(flat)
    print(json.dumps(dict(scope=len(scored), selected=len(chosen), eligible=sum(r['eligible'] for r in scored),
                          markets=dict(Counter(r['market'] for r in chosen)), groups=len(groups)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true', help='Rebuild from saved statements, no requests')
    main(report_only=parser.parse_args().report_only)
