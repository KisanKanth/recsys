import pandas as pd
import numpy as np
import random
import math
from tensorflow import keras
from tensorflow.keras import layers

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
c2 = int(0.9 * len(tr))
tr2 = tr[:c2]
va = tr[c2:]

uid = {}
mid = {}
for i in range(len(us)):
    if us[i] not in uid:
        uid[us[i]] = len(uid)
    if ms[i] not in mid:
        mid[ms[i]] = len(mid)

tot = 0
for i in tr2:
    tot += rs[i]
gm = tot / len(tr2)

sets = []
for rows in [tr2, va, te]:
    a = []
    b = []
    c = []
    for i in rows:
        a.append(uid[us[i]])
        b.append(mid[ms[i]])
        c.append(rs[i] - gm)
    sets.append((np.array(a), np.array(b), np.array(c, dtype="float32")))
xu, xm, y = sets[0]
vu, vm, vy = sets[1]
tu, tm, ty = sets[2]

nu = len(uid)
nm = len(mid)
d = 16
reg = keras.regularizers.l2(1e-5)

iu = keras.Input(shape=(1,))
im = keras.Input(shape=(1,))
eu = layers.Flatten()(layers.Embedding(nu, d, embeddings_regularizer=reg)(iu))
em = layers.Flatten()(layers.Embedding(nm, d, embeddings_regularizer=reg)(im))
bu = layers.Flatten()(layers.Embedding(nu, 1, embeddings_initializer="zeros")(iu))
bm = layers.Flatten()(layers.Embedding(nm, 1, embeddings_initializer="zeros")(im))
dot = layers.Dot(axes=1)([eu, em])
out = layers.Add()([dot, bu, bm])

model = keras.Model([iu, im], out)
model.compile(optimizer=keras.optimizers.Adam(0.005), loss="mse")

stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True)
model.fit([xu, xm], y, validation_data=([vu, vm], vy),
          epochs=30, batch_size=256, callbacks=[stop])

pred = model.predict([tu, tm]).flatten()
e = 0
for j in range(len(ty)):
    p = pred[j] + gm
    if p < 0.5:
        p = 0.5
    if p > 5:
        p = 5
    t = ty[j] + gm
    e += (t - p) * (t - p)
print("test rmse:", math.sqrt(e / len(ty)))