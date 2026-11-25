"""
fc_gen.py - FreshCart synthetic data generator, OLTP loader and star-schema ETL (SQLite).
Method: seeded NumPy generation (seed=42); order sizes 1-5 lines; Zipf-like product popularity;
weekend uplift of 35%; 60% of orders without promotion; 4% of lines returned.
"""
import numpy as np, sqlite3, time, os, datetime as dt

SEED = 42
START, END = dt.date(2025, 1, 1), dt.date(2026, 11, 5)
DEPTS = {
 "Grocery & Staples": ["Rice & Grains", "Pulses", "Cooking Oils", "Spices & Masalas", "Flours"],
 "Beverages": ["Tea", "Coffee", "Soft Drinks", "Juices", "Water"],
 "Dairy & Eggs": ["Milk", "Curd & Yogurt", "Cheese", "Butter & Ghee", "Eggs"],
 "Fresh Produce": ["Fruits", "Vegetables", "Leafy Greens", "Herbs", "Exotic Produce"],
 "Bakery": ["Bread", "Cakes", "Biscuits", "Buns & Rolls", "Cookies"],
 "Frozen Foods": ["Frozen Veg", "Frozen Snacks", "Ice Cream", "Frozen Meals", "Frozen Desserts"],
 "Snacks & Confectionery": ["Chips", "Namkeen", "Chocolates", "Candy", "Dry Fruits"],
 "Personal Care": ["Hair Care", "Skin Care", "Oral Care", "Bath & Body", "Deodorants"],
 "Home Care": ["Detergents", "Dishwash", "Floor Cleaners", "Air Fresheners", "Tissue Paper"],
 "Baby Care": ["Baby Food", "Diapers", "Baby Wipes", "Baby Skincare", "Feeding Needs"],
 "Meat & Seafood": ["Chicken", "Mutton", "Fish", "Prawns", "Cold Cuts"],
 "Household Essentials": ["Batteries", "Light Bulbs", "Kitchenware", "Storage", "Stationery"]}
GEO = {  # region -> state -> cities
 "South": {"Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Tirunelveli", "Erode", "Vellore", "Thanjavur", "Tiruppur", "Dindigul", "Kanchipuram"],
           "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam", "Secunderabad", "Nalgonda", "Mahbubnagar", "Adilabad", "Siddipet", "Suryapet", "Ramagundam"]},
 "West": {"Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Thane", "Solapur", "Kolhapur", "Amravati", "Sangli", "Jalgaon", "Akola"],
          "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Gandhinagar", "Anand", "Bharuch", "Junagadh"]},
 "North": {"Delhi NCR": ["New Delhi", "Gurugram", "Noida", "Faridabad", "Ghaziabad", "Dwarka", "Rohini", "Saket", "Karol Bagh", "Janakpuri", "Pitampura"],
           "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Mohali", "Hoshiarpur", "Pathankot", "Moga", "Firozpur", "Kapurthala", "Barnala", "Sangrur"]}}
SEGMENTS = ["Premium", "Regular", "Budget", "Family", "Student"]
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
REASONS = ["Damaged", "Expired", "Wrong item", "Quality", "Changed mind"]
N_BRANDS, N_PRODUCTS, N_CUST, N_STORES, N_PROMO = 800, 50000, 20000, 120, 40

OLTP_DDL = """
CREATE TABLE regions(region_id INTEGER PRIMARY KEY, region_name TEXT);
CREATE TABLE states(state_id INTEGER PRIMARY KEY, state_name TEXT, region_id INT REFERENCES regions);
CREATE TABLE cities(city_id INTEGER PRIMARY KEY, city_name TEXT, state_id INT REFERENCES states);
CREATE TABLE stores(store_id INTEGER PRIMARY KEY, store_code TEXT, store_name TEXT, store_type TEXT, city_id INT REFERENCES cities);
CREATE TABLE departments(department_id INTEGER PRIMARY KEY, department_name TEXT);
CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT, department_id INT REFERENCES departments);
CREATE TABLE brands(brand_id INTEGER PRIMARY KEY, brand_name TEXT, category_id INT REFERENCES categories);
CREATE TABLE products(product_id INTEGER PRIMARY KEY, sku TEXT, product_name TEXT, pack_size TEXT, unit_price REAL, brand_id INT REFERENCES brands);
CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, customer_name TEXT, segment TEXT, loyalty_tier TEXT, city_id INT);
CREATE TABLE promotions(promo_id INTEGER PRIMARY KEY, promo_name TEXT, promo_type TEXT, discount_pct REAL);
CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INT REFERENCES customers, store_id INT REFERENCES stores, order_date TEXT, promo_id INT REFERENCES promotions);
CREATE TABLE order_lines(order_id INT REFERENCES orders, line_no INT, product_id INT REFERENCES products, qty INT, unit_price REAL, discount REAL, PRIMARY KEY(order_id,line_no));
CREATE TABLE returns(return_id INTEGER PRIMARY KEY, order_id INT, line_no INT, return_date TEXT, return_qty INT, refund_amount REAL, reason TEXT);
"""
OLTP_IDX = """CREATE INDEX ix_o_date ON orders(order_date); CREATE INDEX ix_o_store ON orders(store_id);
CREATE INDEX ix_o_cust ON orders(customer_id); CREATE INDEX ix_ol_prod ON order_lines(product_id);"""

STAR_DDL = """
CREATE TABLE dim_date(date_key INTEGER PRIMARY KEY, full_date TEXT, day_of_week INT, day_name TEXT, is_weekend INT, month_no INT, month_name TEXT,
  year_month TEXT, quarter_no INT, year_quarter TEXT, year INT);
CREATE TABLE dim_product(product_key INTEGER PRIMARY KEY AUTOINCREMENT, product_id INT, sku TEXT, product_name TEXT, pack_size TEXT, brand_name TEXT, category_name TEXT, department_name TEXT);
CREATE TABLE dim_store(store_key INTEGER PRIMARY KEY AUTOINCREMENT, store_id INT, store_code TEXT, store_name TEXT, store_type TEXT, city TEXT, state TEXT, region TEXT);
CREATE TABLE dim_customer(customer_key INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INT, customer_name TEXT, segment TEXT, loyalty_tier TEXT);
CREATE TABLE dim_promotion(promo_key INTEGER PRIMARY KEY AUTOINCREMENT, promo_id INT, promo_name TEXT, promo_type TEXT, discount_pct REAL);
CREATE TABLE dim_return_reason(reason_key INTEGER PRIMARY KEY AUTOINCREMENT, reason TEXT);
CREATE TABLE fact_sales(date_key INT, product_key INT, store_key INT, customer_key INT, promo_key INT, order_number INT, line_number INT,
  quantity INT, unit_price REAL, gross_amount REAL, discount_amount REAL, net_amount REAL, PRIMARY KEY(order_number,line_number));
CREATE TABLE fact_returns(date_key INT, product_key INT, store_key INT, customer_key INT, reason_key INT, order_number INT, line_number INT,
  return_quantity INT, refund_amount REAL);
"""


def build_oltp(path, N, log=print):
    """Generate N sales lines and load the normalised OLTP database. Returns dict of timings."""
    rng = np.random.default_rng(SEED)
    if os.path.exists(path): os.remove(path)
    con = sqlite3.connect(path); con.executescript(OLTP_DDL)
    t0 = time.time()
    # geography
    reg, st, ci, stores = [], [], [], []
    for ri, (r, states) in enumerate(GEO.items(), 1):
        reg.append((ri, r))
    sid = cid = 0
    city_state = []
    for ri, (r, states) in enumerate(GEO.items(), 1):
        for s, cities in states.items():
            sid += 1; st.append((sid, s, ri))
            for c in cities:
                cid += 1; ci.append((cid, c, sid))
    cnt = {c[0]: 0 for c in ci}
    order_cities = [c[0] for c in ci] + [int(x) for x in rng.choice([c[0] for c in ci][:40], N_STORES - len(ci))]
    for i, cc in enumerate(order_cities, 1):
        cnt[cc] += 1; cname = ci[cc - 1][1]
        stores.append((i, f"{cname}-{cnt[cc]:02d}", f"FreshCart {cname} {cnt[cc]:02d}", rng.choice(["Hyper", "Super", "Express"]).item(), cc))
    deps, cats, brands = [], [], []
    for di, (d, cl) in enumerate(DEPTS.items(), 1):
        deps.append((di, d))
        for c in cl: cats.append((len(cats) + 1, c, di))
    for b in range(1, N_BRANDS + 1):
        brands.append((b, f"Brand {b:03d}", (b - 1) % 60 + 1))
    price = np.round(np.exp(rng.normal(4.6, 0.9, N_PRODUCTS)).clip(10, 3000), 2)
    pb = rng.integers(1, N_BRANDS + 1, N_PRODUCTS)
    packs = rng.choice(["250g", "500g", "1kg", "1L", "5kg", "pack of 6", "single"], N_PRODUCTS)
    prods = [(i + 1, f"SKU{i + 1:06d}", f"Product {i + 1:05d}", packs[i].item(), price[i].item(), int(pb[i])) for i in range(N_PRODUCTS)]
    fn = ["Rajesh", "Anita", "Meena", "Vikram", "Suresh", "Priya", "Karthik", "Divya", "Arun", "Lakshmi", "Imran", "Neha", "Manoj", "Sunita"]
    ln = ["Kumar", "Joseph", "Rao", "Sharma", "Reddy", "Iyer", "Singh", "Patel", "Nair", "Khan", "Das", "Mehta"]
    cust = [(i, f"{fn[i % 14]} {ln[(i * 7) % 12]} {i}", SEGMENTS[int(rng.integers(0, 5))], TIERS[int(rng.choice(4, p=[.5, .3, .15, .05]))], int(rng.integers(1, 71))) for i in range(1, N_CUST + 1)]
    ptype = ["Percent off", "BOGO", "Bundle", "Loyalty bonus"]
    pct = np.round(rng.choice([0.05, 0.10, 0.15, 0.20], N_PROMO), 2)
    promos = [(i, f"Promo {i:02d}", ptype[i % 4], pct[i - 1].item()) for i in range(1, N_PROMO + 1)]
    # orders / lines
    lpo = rng.integers(1, 6, N // 2 + 10); cum = np.cumsum(lpo); k = int(np.searchsorted(cum, N)) + 1
    lpo = lpo[:k].copy(); lpo[-1] -= cum[k - 1] - N
    oid_line = np.repeat(np.arange(1, k + 1), lpo)
    starts = np.concatenate([[0], np.cumsum(lpo)[:-1]]); line_no = np.arange(N) - np.repeat(starts, lpo) + 1
    days = (END - START).days + 1
    dts = np.array([START + dt.timedelta(d) for d in range(days)])
    w = np.array([1.35 if d.weekday() >= 5 else 1.0 for d in dts]); w /= w.sum()
    odate_idx = rng.choice(days, k, p=w)
    ostore = rng.choice(N_STORES, k, p=(lambda x: x / x.sum())(rng.uniform(0.5, 1.5, N_STORES))) + 1
    ocust = rng.integers(1, N_CUST + 1, k)
    opromo = np.where(rng.random(k) < 0.4, rng.integers(1, N_PROMO + 1, k), 0)
    ds = [d.isoformat() for d in dts]
    orders = list(zip(range(1, k + 1), ocust.tolist(), ostore.tolist(), [ds[i] for i in odate_idx], [p if p else None for p in opromo.tolist()]))
    lp = ((rng.random(N) ** 1.6) * N_PRODUCTS).astype(int)
    qty = rng.choice([1, 2, 3, 4, 5, 6], N, p=[.4, .25, .15, .1, .06, .04])
    up = price[lp]
    pr = np.concatenate([[0.0], pct])[opromo[oid_line - 1]]
    disc = np.round(qty * up * pr, 2)
    lines = list(zip(oid_line.tolist(), line_no.tolist(), (lp + 1).tolist(), qty.tolist(), up.tolist(), disc.tolist()))
    rmask = np.where(rng.random(N) < 0.04)[0]
    rq = np.array([rng.integers(1, qty[i] + 1) for i in rmask]) if len(rmask) else np.array([], int)
    rets = []
    for j, i in enumerate(rmask):
        d = min(odate_idx[oid_line[i] - 1] + int(rng.integers(1, 15)), days - 1)
        rets.append((j + 1, int(oid_line[i]), int(line_no[i]), ds[d], int(rq[j]), round(float(up[i] * rq[j]), 2), REASONS[int(rng.integers(0, 5))]))
    ins = lambda t, rows, n: con.executemany(f"INSERT INTO {t} VALUES({','.join('?' * n)})", rows)
    ins("regions", reg, 2); ins("states", st, 3); ins("cities", ci, 3); ins("stores", stores, 5); ins("departments", deps, 2)
    ins("categories", cats, 3); ins("brands", brands, 3); ins("products", prods, 6); ins("customers", cust, 5); ins("promotions", promos, 4)
    ins("orders", orders, 5); ins("order_lines", lines, 6); ins("returns", rets, 7)
    con.executescript(OLTP_IDX); con.commit(); con.execute("ANALYZE"); con.commit()
    t = time.time() - t0
    n = {x: con.execute(f"select count(*) from {x}").fetchone()[0] for x in ["orders", "order_lines", "returns"]}
    con.close()
    return {"oltp_load_s": round(t, 2), **n, "oltp_mb": round(os.path.getsize(path) / 1e6, 2)}


def build_star(oltp, path):
    """ETL: dimensions first (surrogate keys), then facts via surrogate-key lookups. Returns timings."""
    if os.path.exists(path): os.remove(path)
    con = sqlite3.connect(path); con.executescript(STAR_DDL)
    con.execute(f"ATTACH '{oltp}' AS src"); T = {}
    t = time.time()
    rows = []
    d = dt.date(2025, 1, 1)
    while d <= dt.date(2026, 12, 31):
        q = (d.month - 1) // 3 + 1
        rows.append((int(d.strftime("%Y%m%d")), d.isoformat(), d.weekday(), d.strftime("%A"), int(d.weekday() >= 5), d.month, d.strftime("%B"), d.strftime("%Y-%m"), q, f"{d.year}-Q{q}", d.year))
        d += dt.timedelta(1)
    con.executemany("INSERT INTO dim_date VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.execute("""INSERT INTO dim_product(product_id,sku,product_name,pack_size,brand_name,category_name,department_name)
      SELECT p.product_id,p.sku,p.product_name,p.pack_size,b.brand_name,c.category_name,d.department_name FROM src.products p
      JOIN src.brands b ON p.brand_id=b.brand_id JOIN src.categories c ON b.category_id=c.category_id JOIN src.departments d ON c.department_id=d.department_id""")
    con.execute("""INSERT INTO dim_store(store_id,store_code,store_name,store_type,city,state,region)
      SELECT s.store_id,s.store_code,s.store_name,s.store_type,ci.city_name,st.state_name,r.region_name FROM src.stores s
      JOIN src.cities ci ON s.city_id=ci.city_id JOIN src.states st ON ci.state_id=st.state_id JOIN src.regions r ON st.region_id=r.region_id""")
    con.execute("INSERT INTO dim_customer(customer_id,customer_name,segment,loyalty_tier) SELECT customer_id,customer_name,segment,loyalty_tier FROM src.customers")
    con.execute("INSERT INTO dim_promotion(promo_id,promo_name,promo_type,discount_pct) VALUES(0,'No Promotion','None',0)")
    con.execute("INSERT INTO dim_promotion(promo_id,promo_name,promo_type,discount_pct) SELECT promo_id,promo_name,promo_type,discount_pct FROM src.promotions")
    con.executemany("INSERT INTO dim_return_reason(reason) VALUES(?)", [(r,) for r in REASONS])
    for s in ["CREATE UNIQUE INDEX ux_p ON dim_product(product_id)", "CREATE UNIQUE INDEX ux_s ON dim_store(store_id)",
              "CREATE UNIQUE INDEX ux_c ON dim_customer(customer_id)", "CREATE UNIQUE INDEX ux_pr ON dim_promotion(promo_id)"]:
        con.execute(s)
    con.commit(); T["dim_load_s"] = round(time.time() - t, 2)
    t = time.time()
    con.execute("""INSERT INTO fact_sales SELECT CAST(replace(o.order_date,'-','') AS INT), dp.product_key, ds.store_key, dc.customer_key, dpr.promo_key,
      o.order_id, ol.line_no, ol.qty, ol.unit_price, ol.qty*ol.unit_price, ol.discount, ol.qty*ol.unit_price-ol.discount
      FROM src.order_lines ol JOIN src.orders o ON ol.order_id=o.order_id JOIN dim_product dp ON dp.product_id=ol.product_id
      JOIN dim_store ds ON ds.store_id=o.store_id JOIN dim_customer dc ON dc.customer_id=o.customer_id
      JOIN dim_promotion dpr ON dpr.promo_id=COALESCE(o.promo_id,0)""")
    con.execute("""INSERT INTO fact_returns SELECT CAST(replace(r.return_date,'-','') AS INT), dp.product_key, ds.store_key, dc.customer_key, rr.reason_key,
      r.order_id, r.line_no, r.return_qty, r.refund_amount FROM src.returns r JOIN src.order_lines ol ON ol.order_id=r.order_id AND ol.line_no=r.line_no
      JOIN src.orders o ON o.order_id=r.order_id JOIN dim_product dp ON dp.product_id=ol.product_id JOIN dim_store ds ON ds.store_id=o.store_id
      JOIN dim_customer dc ON dc.customer_id=o.customer_id JOIN dim_return_reason rr ON rr.reason=r.reason""")
    con.commit(); T["fact_load_s"] = round(time.time() - t, 2)
    con.execute("ANALYZE"); con.commit()
    T["fact_rows"] = con.execute("select count(*) from fact_sales").fetchone()[0]
    con.execute("DETACH src"); con.close()
    T["star_mb"] = round(os.path.getsize(path) / 1e6, 2)
    return T
