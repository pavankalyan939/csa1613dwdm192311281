import sqlite3, shutil, time, json, numpy as np, pandas as pd, datetime as dt
SRC = "/tmp/fcdb/dw_1000000.db"
rng = np.random.default_rng(11)
R = {}


def fresh(name):
    p = f"/tmp/fcdb/{name}.db"; shutil.copy(SRC, p); return sqlite3.connect(p), p


# ---------- E1: how wrong is the ORIGINAL (Type-1) design for 'tier at time of order'? ----------
con, _ = fresh("e1")
cust = pd.read_sql("select customer_key, loyalty_tier from dim_customer", con)
tiers = ["Bronze", "Silver", "Gold", "Platinum"]
chg = cust[cust.loyalty_tier != "Bronze"].sample(frac=0.30, random_state=1).copy()   # 30 % of Silver/Gold/Platinum customers were upgraded during the period
chg["change_date"] = [int((dt.date(2026, 1, 1) + dt.timedelta(int(x))).strftime("%Y%m%d")) for x in rng.integers(0, 300, len(chg))]
chg["old_tier"] = [tiers[tiers.index(t) - 1] for t in chg.loyalty_tier]
f = pd.read_sql("select customer_key, date_key, net_amount from fact_sales", con)
f = f.merge(cust, on="customer_key").merge(chg[["customer_key", "change_date", "old_tier"]], on="customer_key", how="left")
f["true_tier"] = np.where(f.change_date.notna() & (f.date_key < f.change_date), f.old_tier, f.loyalty_tier)
t1 = f.groupby("loyalty_tier").net_amount.sum(); tr = f.groupby("true_tier").net_amount.sum()
wrong = f.loc[f.loyalty_tier != f.true_tier, "net_amount"].sum()
R["e1"] = {"tiers": {t: [round(float(t1[t]), 0), round(float(tr[t]), 0)] for t in tiers}, "wrong_pct": round(100 * wrong / f.net_amount.sum(), 2),
           "wrong_rows": int((f.loyalty_tier != f.true_tier).sum()), "changed_customers": len(chg)}
con.close()

# ---------- build revised design on copy A (indexed) and copy B (no fact indexes) ----------
def revise(con):
    c = con
    c.executescript("""
    CREATE TABLE dim_channel(channel_key INTEGER PRIMARY KEY, channel_name TEXT); INSERT INTO dim_channel VALUES(0,'Store'),(1,'Online');
    CREATE TABLE dim_delivery_partner(partner_key INTEGER PRIMARY KEY, partner_name TEXT); INSERT INTO dim_delivery_partner VALUES(0,'Not applicable'),(1,'SwiftDrop'),(2,'QuickKart'),(3,'RoadRunner'),(4,'ShipEasy'),(5,'LocalGo');
    ALTER TABLE fact_sales ADD COLUMN channel_key INT DEFAULT 0;
    ALTER TABLE fact_sales ADD COLUMN delivery_partner_key INT DEFAULT 0;
    CREATE TABLE product_xref(source_system TEXT, source_code TEXT, product_key INT, PRIMARY KEY(source_system, source_code));
    INSERT INTO product_xref SELECT 'ONLINE', printf('ON%07d', product_id), product_key FROM dim_product;
    CREATE TABLE dim_customer2(customer_key INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INT, customer_name TEXT, segment TEXT, loyalty_tier TEXT, address TEXT,
        eff_from INT, eff_to INT, is_current INT);
    INSERT INTO dim_customer2(customer_key,customer_id,customer_name,segment,loyalty_tier,address,eff_from,eff_to,is_current)
        SELECT customer_key, customer_id, customer_name, segment, loyalty_tier, 'Addr-'||customer_id||'-v1', 20250101, 99991231, 1 FROM dim_customer;
    CREATE INDEX ix_c2 ON dim_customer2(customer_id, eff_from);
    CREATE TABLE etl_audit(batch_id TEXT, src_rows INT, loaded INT, rejected INT, src_amount REAL, loaded_amount REAL);""")
    c.commit()


def make_stage(c, n_orders, day, seed):
    r = np.random.default_rng(seed); lpo = r.integers(1, 6, n_orders)
    oid = np.repeat(np.arange(n_orders), lpo) + int(seed) * 100000 + 10_000_000
    N = len(oid); lno = np.concatenate([np.arange(1, k + 1) for k in lpo])
    custs = np.repeat(r.integers(1, 20001, n_orders), lpo); pid = (r.random(N) ** 1.6 * 50000).astype(int) + 1
    prices = dict(c.execute("select product_id, 1 from dim_product").fetchall())  # existence only
    up = np.round(np.exp(r.normal(4.6, 0.9, N)).clip(10, 3000), 2); q = r.integers(1, 5, N)
    stores = np.repeat(r.integers(1, 121, n_orders), lpo); part = np.repeat(r.integers(1, 6, n_orders), lpo)
    rows = list(zip(oid.tolist(), lno.tolist(), custs.tolist(), [f"ON{p:07d}" for p in pid], stores.tolist(), q.tolist(), up.tolist(), [day] * N, part.tolist()))
    c.execute("DROP TABLE IF EXISTS stg"); c.execute("CREATE TABLE stg(order_id INT, line_no INT, customer_id INT, prod_code TEXT, store_id INT, qty INT, unit_price REAL, date_key INT, partner INT)")
    c.executemany("INSERT INTO stg VALUES(?,?,?,?,?,?,?,?,?)", rows); c.commit(); return N


def scd2(c, day, batch, pct=0.05):
    ids = c.execute("select distinct customer_id from stg").fetchall(); k = max(1, int(len(ids) * pct))
    ch = [ids[i][0] for i in np.random.default_rng(batch).choice(len(ids), k, replace=False)]
    for cid in ch:
        c.execute("UPDATE dim_customer2 SET eff_to=?, is_current=0 WHERE customer_id=? AND is_current=1", (day - 1, cid))
        c.execute("INSERT INTO dim_customer2(customer_id,customer_name,segment,loyalty_tier,address,eff_from,eff_to,is_current) "
                  "SELECT customer_id,customer_name,segment,CASE loyalty_tier WHEN 'Bronze' THEN 'Silver' ELSE 'Gold' END,'Addr-'||customer_id||'-v'||?,?,99991231,1 "
                  "FROM dim_customer2 WHERE customer_id=? AND eff_to=?", (batch + 2, day, cid, day - 1))
    c.commit(); return k


def load_fact(c, batch_id):
    t = time.perf_counter()
    c.execute("""INSERT INTO fact_sales(date_key,product_key,store_key,customer_key,promo_key,order_number,line_number,quantity,unit_price,gross_amount,discount_amount,net_amount,channel_key,delivery_partner_key)
      SELECT s.date_key, x.product_key, ds.store_key, dc.customer_key, 1, s.order_id, s.line_no, s.qty, s.unit_price, s.qty*s.unit_price, 0, s.qty*s.unit_price, 1, s.partner
      FROM stg s JOIN product_xref x ON x.source_system='ONLINE' AND x.source_code=s.prod_code JOIN dim_store ds ON ds.store_id=s.store_id
      JOIN dim_customer2 dc ON dc.customer_id=s.customer_id AND s.date_key BETWEEN dc.eff_from AND dc.eff_to""")
    c.commit(); return time.perf_counter() - t


def refresh_agg(c, day):
    t = time.perf_counter(); m0, m1 = day // 100 * 100 + 1, day // 100 * 100 + 31
    c.execute("DELETE FROM agg_day_store WHERE date_key=?", (day,))
    c.execute("INSERT INTO agg_day_store SELECT date_key, store_key, SUM(net_amount) FROM fact_sales WHERE date_key=? GROUP BY 1,2", (day,))
    c.execute("DELETE FROM agg_day_dept WHERE date_key=?", (day,))
    c.execute("INSERT INTO agg_day_dept SELECT f.date_key, d.is_weekend, p.department_name, SUM(f.net_amount) FROM fact_sales f JOIN dim_date d ON d.date_key=f.date_key JOIN dim_product p ON p.product_key=f.product_key WHERE f.date_key=? GROUP BY 1,2,3", (day,))
    ym = f"{day // 10000}-{(day // 100) % 100:02d}"
    c.execute("DELETE FROM agg_month_store_cat WHERE year_month=?", (ym,))
    c.execute("INSERT INTO agg_month_store_cat SELECT d.year, d.year_quarter, d.year_month, f.store_key, p.category_name, SUM(f.net_amount), SUM(f.quantity) FROM fact_sales f JOIN dim_date d ON d.date_key=f.date_key JOIN dim_product p ON p.product_key=f.product_key WHERE f.date_key BETWEEN ? AND ? GROUP BY 1,2,3,4,5", (m0, m1))
    c.commit(); return time.perf_counter() - t


def full_rebuild(c):
    import fc_bench
    t = time.perf_counter()
    for n in ["agg_month_store_cat", "agg_day_dept", "agg_day_store"]: c.execute(f"DROP TABLE {n}")
    for s in fc_bench.AGG_SQL + fc_bench.AGG_IDX: c.execute(s)
    c.commit(); return time.perf_counter() - t


DAY = 20261106
# Copy A: indexed fact table, 3 hourly micro-batches then a whole-day batch
A, pa = fresh("e2a"); revise(A); out = {"hourly": []}
for h in range(3):
    n = make_stage(A, 833, DAY, h + 1); k = scd2(A, DAY, h + 1); tl = load_fact(A, h); ta = refresh_agg(A, DAY)
    out["hourly"].append({"lines": n, "scd2_changes": k, "fact_s": round(tl, 3), "agg_s": round(ta, 3)})
out["full_rebuild_s_1M"] = round(full_rebuild(A), 2)
n = make_stage(A, 20000, DAY + 1, 9); k = scd2(A, DAY + 1, 9); out["day_indexed"] = {"lines": n, "scd2": k, "fact_s": round(load_fact(A, 9), 2), "agg_s": round(refresh_agg(A, DAY + 1), 2)}
out["rows_after"] = A.execute("select count(*) from fact_sales").fetchone()[0]
out["dim_c2_rows"] = A.execute("select count(*) from dim_customer2").fetchone()[0]
# audit & reconciliation for the last batch
src_amt = A.execute("select round(sum(qty*unit_price),2), count(*) from stg").fetchone()
ld = A.execute("select round(sum(net_amount),2), count(*) from fact_sales where date_key=?", (DAY + 1,)).fetchone()
out["audit"] = {"stg_rows": src_amt[1], "stg_amount": src_amt[0], "fact_rows": ld[1], "fact_amount": ld[0]}
# dashboard query on agg after refresh
t = time.perf_counter(); A.execute("SELECT st.region, a.category_name, a.year_quarter, SUM(a.net_amount) FROM agg_month_store_cat a JOIN dim_store st ON a.store_key=st.store_key WHERE a.year=2026 GROUP BY 1,2,3").fetchall()
out["dash_q1_agg_ms"] = round((time.perf_counter() - t) * 1000, 1)
t = time.perf_counter(); A.execute("SELECT st.region, p.category_name, d.year_quarter, SUM(f.net_amount) FROM fact_sales f JOIN dim_date d ON f.date_key=d.date_key JOIN dim_product p ON f.product_key=p.product_key JOIN dim_store st ON f.store_key=st.store_key WHERE d.year=2026 GROUP BY 1,2,3").fetchall()
out["dash_q1_star_ms"] = round((time.perf_counter() - t) * 1000, 1)
A.close()
# Copy B: fact indexes dropped before the day batch (index-maintenance cost)
B, pb = fresh("e2b"); revise(B)
for ix in ["ix_f_date", "ix_f_store", "ix_f_prod", "ix_f_cust", "ix_f_promo", "ix_f_cover"]: B.execute(f"DROP INDEX {ix}")
B.commit(); n = make_stage(B, 20000, DAY + 1, 9); scd2(B, DAY + 1, 9); out["day_noindex"] = {"lines": n, "fact_s": round(load_fact(B, 9), 2)}
# range-join (SCD2) vs equality-join (Type 1) cost on the 1M historical fact reload
t = time.perf_counter(); B.execute("SELECT count(*) FROM fact_sales f JOIN dim_customer dc ON dc.customer_key=f.customer_key").fetchone(); out["join_type1_s"] = round(time.perf_counter() - t, 2)
B.execute("UPDATE fact_sales SET date_key=date_key WHERE 0")
t = time.perf_counter(); B.execute("SELECT count(*) FROM fact_sales f JOIN dim_customer2 dc ON dc.customer_id=(SELECT customer_id FROM dim_customer WHERE customer_key=f.customer_key) AND f.date_key BETWEEN dc.eff_from AND dc.eff_to").fetchone(); out["join_scd2_s"] = round(time.perf_counter() - t, 2)
B.close()
R["e2"] = out
json.dump(R, open("/tmp/fcdb/exp11.json", "w"), indent=1); print(json.dumps(R, indent=1))
