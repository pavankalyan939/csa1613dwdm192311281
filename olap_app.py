"""
olap_app.py - Interactive OLAP navigator for the FreshCart star schema (SQLite + pandas).
Usage:  python olap_app.py dw_100000.db oltp_100000.db
The user chooses the operation, dimension, level, measure and filter values through input prompts.
The session keeps a current 'view' (one level per dimension kept in the cube), so operations chain like a real OLAP client.
"""
import sqlite3, sys
import pandas as pd

H = {"time": ["day", "month", "quarter", "year"], "product": ["product", "category", "department"],
     "store": ["store", "city", "state"], "segment": ["segment"]}
COL = {"time": {"day": "d.full_date", "month": "d.year_month", "quarter": "d.year_quarter", "year": "d.year"},
       "product": {"product": "p.product_name", "category": "p.category_name", "department": "p.department_name"},
       "store": {"store": "s.store_name", "city": "s.city", "state": "s.state"}, "segment": {"segment": "c.segment"}}
JOIN = {"time": "JOIN dim_date d ON f.date_key=d.date_key", "product": "JOIN dim_product p ON f.product_key=p.product_key",
        "store": "JOIN dim_store s ON f.store_key=s.store_key", "segment": "JOIN dim_customer c ON f.customer_key=c.customer_key"}
MEAS = {"net_sales": "SUM(f.net_amount)", "quantity": "SUM(f.quantity)", "gross_sales": "SUM(f.gross_amount)"}
pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:,.2f}".format)


def cube(con, view, filters, measure="net_sales"):
    """view: {dim: level}; filters: [(dim, level, [values])]. Builds and runs the aggregate query on the star schema."""
    dims = list(dict.fromkeys(list(view) + [f[0] for f in filters]))
    sel = [f"{COL[d][l]} AS {d}_{l}" for d, l in view.items()]
    where, params = [], []
    for d, l, vals in filters:
        where.append(f"{COL[d][l]} IN ({','.join('?' * len(vals))})"); params += vals
    sql = f"SELECT {', '.join(sel + [MEAS[measure] + ' AS ' + measure])} FROM fact_sales f {' '.join(JOIN[d] for d in dims)}"
    if where: sql += " WHERE " + " AND ".join(where)
    if sel: sql += f" GROUP BY {', '.join(str(i + 1) for i in range(len(sel)))} ORDER BY {', '.join(str(i + 1) for i in range(len(sel)))}"
    return pd.read_sql_query(sql, con, params=params), sql


def cuboid_name(view):
    return "(" + ", ".join(view.get(d, "ALL") for d in H) + ")"


def pick(prompt, options):
    for i, o in enumerate(options, 1): print(f"   {i}. {o}")
    while True:
        s = input(prompt).strip()
        if s.isdigit() and 1 <= int(s) <= len(options): return options[int(s) - 1]
        if s in options: return s
        print("   invalid choice")


def values_for(con, d, l):
    t = {"time": "dim_date d", "product": "dim_product p", "store": "dim_store s", "segment": "dim_customer c"}[d]
    return [str(r[0]) for r in con.execute(f"SELECT DISTINCT {COL[d][l]} FROM {t} ORDER BY 1 LIMIT 12")]


def show(df, sql, view, filters):
    print(f"\n   Cuboid used : {cuboid_name(view)}   filters: {[(d, l, v) for d, l, v in filters] or 'none'}")
    print(f"   SQL         : {sql}\n   rows={len(df)}   grand total of result = {df.iloc[:, -1].sum():,.2f}\n")
    print(df.head(15).to_string(index=False)); print("   ..." if len(df) > 15 else "")


def reconcile(con, oltp, measure="net_sales"):
    star_total = con.execute("SELECT SUM(net_amount) FROM fact_sales").fetchone()[0]
    src = sqlite3.connect(oltp)
    src_total = src.execute("SELECT SUM(qty*unit_price-discount) FROM order_lines").fetchone()[0]
    print(f"\n   [1] Apex cuboid (star)  = {star_total:,.2f}\n       OLTP independent sum = {src_total:,.2f}   diff = {star_total - src_total:,.6f}")
    for d, levels in H.items():
        for l in levels:
            df, _ = cube(con, {d: l}, [], measure)
            print(f"   [2] roll-up check {d}:{l:<10} total = {df.iloc[:, -1].sum():,.2f}  {'OK' if abs(df.iloc[:, -1].sum() - star_total) < 0.01 else 'MISMATCH'}")
    raw = pd.read_sql_query("""SELECT c.category_name AS cat, SUM(ol.qty*ol.unit_price-ol.discount) AS raw FROM order_lines ol JOIN orders o ON o.order_id=ol.order_id
        JOIN products p ON p.product_id=ol.product_id JOIN brands b ON b.brand_id=p.brand_id JOIN categories c ON c.category_id=b.category_id
        WHERE o.order_date>='2026-01-01' GROUP BY 1""", src)
    cb, _ = cube(con, {"product": "category"}, [("time", "year", ["2026"])], measure)
    m = raw.merge(cb, left_on="cat", right_on="product_category")
    print(f"   [3] category totals 2026: {len(m)} categories compared, max abs difference = {(m['raw'] - m[measure]).abs().max():.6f}")
    src.close()


def main(dbp, oltp):
    con = sqlite3.connect(dbp)
    view, filters, measure = {"time": "year", "product": "department"}, [], "net_sales"
    while True:
        print(f"\n=== FreshCart OLAP ===  view={cuboid_name(view)}  filters={len(filters)}  measure={measure}")
        op = pick("Choose operation: ", ["roll-up", "drill-down", "slice", "dice", "pivot", "change measure", "reset view", "reconcile", "quit"])
        if op == "quit": break
        if op == "reset view": view, filters = {"time": "year", "product": "department"}, []
        elif op == "change measure": measure = pick("Measure: ", list(MEAS))
        elif op == "reconcile": reconcile(con, oltp, measure); continue
        elif op in ("roll-up", "drill-down"):
            d = pick("Dimension: ", list(H)); lv = H[d]
            if op == "roll-up":
                if d not in view: print("   dimension not in view"); continue
                i = lv.index(view[d]); view = {k: v for k, v in view.items() if k != d} | ({d: lv[i + 1]} if i + 1 < len(lv) else {})
                view = {k: view[k] for k in H if k in view}   # keep canonical dimension order; top level = ALL removes the dimension
            else:
                if d not in view: view[d] = lv[-1]
                elif lv.index(view[d]) > 0: view[d] = lv[lv.index(view[d]) - 1]
                else: print("   already at the lowest level"); continue
                view = {k: view[k] for k in H if k in view}
        elif op == "slice":
            d = pick("Dimension to slice on: ", list(H)); l = pick("Level: ", H[d])
            print("   sample values:", values_for(con, d, l)); v = input("   Value: ").strip()
            filters = [f for f in filters if f[0] != d] + [(d, l, [v])]; view.pop(d, None)
        elif op == "dice":
            filters = []
            while True:
                d = pick("Dimension to filter (dice): ", list(H)); l = pick("Level: ", H[d])
                print("   sample values:", values_for(con, d, l)); v = [x.strip() for x in input("   Values (comma-separated): ").split(",")]
                filters.append((d, l, v))
                if input("   add another filter? (y/n): ").lower() != "y": break
        elif op == "pivot":
            if len(view) < 2: print("   need two dimensions in view"); continue
            r = pick("Row dimension: ", list(view)); c = pick("Column dimension: ", [x for x in view if x != r])
            df, sql = cube(con, view, filters, measure)
            pv = df.pivot_table(index=f"{r}_{view[r]}", columns=f"{c}_{view[c]}", values=measure, aggfunc="sum", margins=True, margins_name="TOTAL")
            print(f"\n   Cuboid used: {cuboid_name(view)}\n   SQL: {sql}\n"); print(pv.head(15).to_string()); continue
        df, sql = cube(con, view, filters, measure); show(df, sql, view, filters)
    con.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
