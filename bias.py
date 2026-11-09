import pandas as pd
import random
import math

r = pd.read_csv("ratings.csv")
us = r["userId"].tolist()
ms = r["movieId"].tolist()
rs = r["rating"].tolist()

idx = list(range(len(rs)))
random.seed(42)
random.shuffle(idx)
cut = int(0.8 * len(idx))
tr = idx[:cut]
te = idx[cut:]

tot = 0
for i in tr:
    tot += rs[i]
gm = tot / len(tr)

for k in [0, 5, 10, 25]:
    dm = {}
    cm = {}
    for i in tr:
        dm[ms[i]] = dm.get(ms[i], 0) + (rs[i] - gm)
        cm[ms[i]] = cm.get(ms[i], 0) + 1
    bm = {}
    for m in dm:
        bm[m] = dm[m] / (cm[m] + k)

    du = {}
    cu = {}
    for i in tr:
        du[us[i]] = du.get(us[i], 0) + (rs[i] - gm - bm[ms[i]])
        cu[us[i]] = cu.get(us[i], 0) + 1
    bu = {}
    for u in du:
        bu[u] = du[u] / (cu[u] + k)

    e = 0
    for i in te:
        p = gm + bm.get(ms[i], 0) + bu.get(us[i], 0)
        if p < 0.5:
            p = 0.5
        if p > 5:
            p = 5
        e += (rs[i] - p) * (rs[i] - p)
    print("k:", k, "rmse:", math.sqrt(e / len(te)))