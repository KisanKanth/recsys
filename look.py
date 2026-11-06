import pandas as pd

r = pd.read_csv("ratings.csv")
m = pd.read_csv("movies.csv")

print("ratings:", r.shape)
print(r.head())
print("movies:", m.shape)
print(m.head())

us = r["userId"].tolist()
ms = r["movieId"].tolist()
rs = r["rating"].tolist()

uc = {}
mc = {}
rc = {}
for i in range(len(us)):
    uc[us[i]] = uc.get(us[i], 0) + 1
    mc[ms[i]] = mc.get(ms[i], 0) + 1
    rc[rs[i]] = rc.get(rs[i], 0) + 1

nu = len(uc)
nm = len(mc)
tot = len(us)
print("users:", nu)
print("movies rated:", nm)
print("possible pairs:", nu * nm)
print("actual ratings:", tot)
print("filled %:", 100 * tot / (nu * nm))

print("rating counts:")
for k in sorted(rc):
    print(k, rc[k])

for name, d in [("user", uc), ("movie", mc)]:
    lo = 10**9
    hi = 0
    s = 0
    for k in d:
        v = d[k]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        s += v
    print(name, "ratings min/avg/max:", lo, s / len(d), hi)