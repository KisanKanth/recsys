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
msum = {}
mcnt = {}
usum = {}
ucnt = {}
for i in tr:
    tot += rs[i]
    msum[ms[i]] = msum.get(ms[i], 0) + rs[i]
    mcnt[ms[i]] = mcnt.get(ms[i], 0) + 1
    usum[us[i]] = usum.get(us[i], 0) + rs[i]
    ucnt[us[i]] = ucnt.get(us[i], 0) + 1
gm = tot / len(tr)
print("global mean:", gm)

e1 = 0
e2 = 0
e3 = 0
new = 0
for i in te:
    t = rs[i]

    p1 = gm

    if ms[i] in mcnt:
        p2 = msum[ms[i]] / mcnt[ms[i]]
    else:
        p2 = gm
        new += 1

    if us[i] in ucnt:
        p3 = usum[us[i]] / ucnt[us[i]]
    else:
        p3 = gm

    e1 += (t - p1) * (t - p1)
    e2 += (t - p2) * (t - p2)
    e3 += (t - p3) * (t - p3)

n = len(te)
print("train:", len(tr), "test:", n)
print("test movies never seen in train:", new)
print("rmse global mean:", math.sqrt(e1 / n))
print("rmse movie mean:", math.sqrt(e2 / n))
print("rmse user mean:", math.sqrt(e3 / n))