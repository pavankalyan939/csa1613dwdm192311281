import sqlite3, time, os, re, json, sys, statistics
import fc_gen

OLTP_Q = {
"Q1": """SELECT r.region_name, c.category_name, strftime('%Y',o.order_date)||'-Q'||((CAST(strftime('%m',o.order_date) AS INT)+2)/3) AS qtr,
 SUM(ol.qty*ol.unit_price-ol.discount) AS net
 FROM order_lines ol JOIN orders o ON ol.order_id=o.order_id JOIN products p ON ol.product_id=p.product_id
 JOIN brands b ON p.brand_id=b.brand_id JOIN categories c ON b.category_id=c.category_id JOIN stores s ON o.store_id=s.store_id
 JOIN cities ci ON s.city_id=ci.city_id JOIN states st ON ci.state_id=st.state_id JOIN regions r ON st.region_id=r.region_id
 WHERE o.order_date>='2026-01-01' GROUP BY 1,2,3""",
"Q2": """SELECT d.department_name, CASE WHEN strftime('%w',o.order_date) IN ('0','6') THEN 'Weekend' ELSE 'Weekday' END AS daytype,
 SUM(ol.qty*ol.unit_price-ol.discount) AS net
 FROM order_lines ol JOIN orders o ON ol.order_id=o.order_id JOIN products p ON ol.product_id=p.product_id
 JOIN brands b ON p.brand_id=b.brand_id JOIN categories c ON b.category_id=c.category_id JOIN departments d ON c.department_id=d.department_id
 WHERE o.order_date>='2026-01-01' GROUP BY 1,2""",
"Q3": """SELECT s.store_name, ci.city_name, st.state_name,
 SUM(CASE WHEN o.order_date BETWEEN '2026-01-01' AND '2026-11-05' THEN ol.qty*ol.unit_price-ol.discount END) AS ytd,
 SUM(CASE WHEN o.order_date BETWEEN '2025-01-01' AND '2025-11-05' THEN ol.qty*ol.unit_price-ol.discount END) AS ly
 FROM order_lines ol JOIN orders o ON ol.order_id=o.order_id JOIN stores s ON o.store_id=s.store_id
 JOIN cities ci ON s.city_id=ci.city_id JOIN states st ON ci.state_id=st.state_id GROUP BY s.store_id"""}
STAR_Q = {
"Q1": """SELECT st.region, p.category_name, d.year_quarter, SUM(f.net_amount) FROM fact_sales f JOIN dim_date d ON f.date_key=d.date_key
 JOIN dim_product p ON f.product_key=p.product_key JOIN dim_store st ON f.store_key=st.store_key WHERE d.year=2026 GROUP BY 1,2,3""",
"Q2": """SELECT p.department_name, CASE d.is_weekend WHEN 1 THEN 'Weekend' ELSE 'Weekday' END, SUM(f.net_amount) FROM fact_sales f
 JOIN dim_date d ON f.date_key=d.date_key JOIN dim_product p ON f.product_key=p.product_key WHERE d.year=2026 GROUP BY 1,2""",
"Q3": """SELECT st.store_name, st.city, st.state,
 SUM(CASE WHEN f.date_key BETWEEN 20260101 AND 20261105 THEN f.net_amount END), SUM(CASE WHEN f.date_key BETWEEN 20250101 AND 20251105 THEN f.net_amount END)
 FROM fact_sales f JOIN dim_store st ON f.store_key=st.store_key GROUP BY f.store_key"""}
AGG_Q = {
"Q1": """SELECT st.region, a.category_name, a.year_quarter, SUM(a.net_amount) FROM agg_month_store_cat a JOIN dim_store st ON a.store_key=st.store_key
 WHERE a.year=2026 GROUP BY 1,2,3""",
"Q2": """SELECT department_name, CASE is_weekend WHEN 1 THEN 'Weekend' ELSE 'Weekday' END, SUM(net_amount) FROM agg_day_dept WHERE date_key>=20260101 GROUP BY 1,2""",
"Q3": """SELECT st.store_name, st.city, st.state, SUM(CASE WHEN a.date_key BETWEEN 20260101 AND 20261105 THEN a.net_amount END),
 SUM(CASE WHEN a.date_key BETWEEN 20250101 AND 20251105 THEN a.net_amount END) FROM agg_day_store a JOIN dim_store st ON a.store_key=st.store_key GROUP BY a.store_key"""}
JOINS = {"OLTP": {"Q1": 8, "Q2": 5, "Q3": 4}, "STAR": {"Q1": 3, "Q2": 2, "Q3": 1}, "AGG": {"Q1": 1, "Q2": 0, "Q3": 1}}

AGG_SQL = ["""CREATE TABLE agg_month_store_cat AS SELECT d.year, d.year_quarter, d.year_month, f.store_key, p.category_name, SUM(f.net_amount) net_amount, SUM(f.quantity) qty
  FROM fact_sales f JOIN dim_date d ON f.date_key=d.date_key JOIN dim_product p ON f.product_key=p.product_key GROUP BY 1,2,3,4,5""",
 """CREATE TABLE agg_day_dept AS SELECT f.date_key, d.is_weekend, p.department_name, SUM(f.net_amount) net_amount FROM fact_sales f
  JOIN dim_date d ON f.date_key=d.date_key JOIN dim_product p ON f.product_key=p.product_key GROUP BY 1,2,3""",
 """CREATE TABLE agg_day_store AS SELECT date_key, store_key, SUM(net_amount) net_amount FROM fact_sales GROUP BY 1,2"""]
IDX_SQL = ["CREATE INDEX ix_f_date ON fact_sales(date_key)", "CREATE INDEX ix_f_store ON fact_sales(store_key)", "CREATE INDEX ix_f_prod ON fact_sales(product_key)",
           "CREATE INDEX ix_f_cust ON fact_sales(customer_key)", "CREATE INDEX ix_f_promo ON fact_sales(promo_key)",
           "CREATE INDEX ix_f_cover ON fact_sales(date_key, product_key, store_key, net_amount)",
           "CREATE INDEX ix_dd_year ON dim_date(year)"]
AGG_IDX = ["CREATE INDEX ix_a1 ON agg_month_store_cat(year)", "CREATE INDEX ix_a2 ON agg_day_dept(date_key)", "CREATE INDEX ix_a3 ON agg_day_store(date_key)"]


BIG = {"order_lines", "orders", "fact_sales", "agg_month_store_cat", "agg_day_dept", "agg_day_store"}
FILT = {  # rows of a big table that the date predicate lets an index/driver actually touch
 "OLTP": {"Q1": {"orders": "SELECT count(*) FROM orders WHERE order_date>='2026-01-01'",
                 "order_lines": "SELECT count(*) FROM order_lines ol JOIN orders o ON o.order_id=ol.order_id WHERE o.order_date>='2026-01-01'"}},
 "STAR": {"Q1": {"fact_sales": "SELECT count(*) FROM fact_sales WHERE date_key>=20260101"}},
 "AGG": {"Q1": {"agg_month_store_cat": "SELECT count(*) FROM agg_month_store_cat WHERE year=2026"}, "Q2": {"agg_day_dept": "SELECT count(*) FROM agg_day_dept WHERE date_key>=20260101"}}}
FILT["OLTP"]["Q2"] = FILT["OLTP"]["Q1"]; FILT["STAR"]["Q2"] = FILT["STAR"]["Q1"]


def run(con, sql, cfg, q, reps=3):
    cnt = [0]
    def cb(): cnt[0] += 1; return 0
    plan = con.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
    alias = {a: t for t, a in re.findall(r"(?:FROM|JOIN)\s+(\w+)\s+(\w+)", sql) if a.upper() not in ("ON", "WHERE", "JOIN", "GROUP")}
    times = []; steps = 0
    for i in range(reps):
        cnt[0] = 0; con.set_progress_handler(cb, 1000)
        t = time.perf_counter(); rows = con.execute(sql).fetchall(); times.append(time.perf_counter() - t)
        con.set_progress_handler(None, 0); steps = cnt[0]
    cfgk = "STAR" if cfg.startswith("STAR") and cfg != "STAR+AGG" else ("AGG" if cfg == "STAR+AGG" else "OLTP")
    f = FILT.get(cfgk, {}).get(q, {})
    rows_read, probes, plantxt, seen, filtered = 0, 0, [], set(), set()
    for p in plan:
        d = p[3]; plantxt.append(d)
        m = re.match(r"(SCAN|SEARCH) (\w+)", d)
        if not m: continue
        tbl = alias.get(m.group(2), m.group(2))
        if m.group(1) == "SCAN":
            rows_read += con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        else:
            probes += 1
            if tbl in BIG and "INTEGER PRIMARY KEY" not in d and tbl not in seen:
                seen.add(tbl)
                date_used = ("date" in d.split("USING", 1)[-1]) or ("year" in d.split("USING", 1)[-1])
                child_of_filtered = tbl == "order_lines" and "orders" in filtered
                if (date_used or child_of_filtered) and tbl in f:
                    filtered.add(tbl); rows_read += con.execute(f[tbl]).fetchone()[0]
                else:
                    rows_read += con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
    return {"time_ms": round(statistics.median(times) * 1000, 1), "result_rows": len(rows), "rows_read": rows_read, "search_steps": probes,
            "vm_k_steps": steps, "plan": plantxt}


def size(p): return round(os.path.getsize(p) / 1e6, 2)


def main(N, d="/tmp/fcdb"):
    os.makedirs(d, exist_ok=True)
    o, s = f"{d}/oltp_{N}.db", f"{d}/dw_{N}.db"
    R = {"N": N}
    R["oltp"] = fc_gen.build_oltp(o, N); R["star_load"] = fc_gen.build_star(o, s)
    co = sqlite3.connect(o); R["res"] = {"OLTP": {k: run(co, v, "OLTP", k) for k, v in OLTP_Q.items()}}; co.close()
    cs = sqlite3.connect(s); R["res"]["STAR"] = {k: run(cs, v, "STAR", k) for k, v in STAR_Q.items()}
    base = size(s)
    t = time.time(); [cs.execute(x) for x in IDX_SQL]; cs.commit(); R["index_build_s"] = round(time.time() - t, 2); a = size(s); R["index_mb"] = round(a - base, 2)
    t = time.time(); [cs.execute(x) for x in AGG_SQL]; [cs.execute(x) for x in AGG_IDX]; cs.commit(); R["agg_build_s"] = round(time.time() - t, 2); R["agg_mb"] = round(size(s) - a, 2)
    cs.execute("ANALYZE"); cs.commit()
    R["agg_rows"] = {t: cs.execute(f"select count(*) from {t}").fetchone()[0] for t in ["agg_month_store_cat", "agg_day_dept", "agg_day_store"]}
    R["res"]["STAR+IDX"] = {k: run(cs, v, "STAR+IDX", k) for k, v in STAR_Q.items()}
    R["res"]["STAR+AGG"] = {k: run(cs, v, "STAR+AGG", k) for k, v in AGG_Q.items()}
    # reconciliation of the three summary tables with the fact table
    tot = cs.execute("select round(sum(net_amount),2) from fact_sales").fetchone()[0]
    R["recon"] = {"fact": tot, **{t: cs.execute(f"select round(sum(net_amount),2) from {t}").fetchone()[0] for t in R["agg_rows"]}}
    R["star_total_mb"] = size(s); cs.close()
    json.dump(R, open(f"{d}/res_{N}.json", "w"), indent=1)
    print(N, "done", R["star_load"], R["index_build_s"], R["agg_build_s"], flush=True)


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]): main(n)
