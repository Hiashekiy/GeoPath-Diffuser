import json
from math import comb
def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)
d = json.load(open("outputs/reports/ablation_transfer_all.json", encoding="utf-8"))
names = [a["arm"] for a in d["arms"]]
print("=== 4 arms, %d samples, %s ===" % (d["samples"], d["safety_mode"]))
for a in d["arms"]:
    print("%-22s free %.3f  coll %.4f  chamfer %.3f  ctrl %.3f  len %.3f"
          % (a["arm"], a["collision_free_rate"], a["collision_rate"]["mean"],
             a["chamfer_m"]["mean"], a["control_rmse_m"]["mean"], a["length_ratio"]["mean"]))
base, recs = names[0], d["records"]
b = {r["index"]: r for r in recs[base]}
for other in names[1:]:
    cur = {r["index"]: r for r in recs[other]}
    common = sorted(set(b) & set(cur))
    parts = []
    for key in ("collision_rate", "chamfer_m", "length_error"):
        w = l = t = 0
        for i in common:
            va, vb = b[i][key], cur[i][key]
            if abs(va - vb) < 1e-12: t += 1
            elif va < vb: w += 1
            else: l += 1
        parts.append("%s %d/%d/%d p=%.3f" % (key, w, l, t, sign_p(w, l)))
    print("  vs %-22s %s" % (other, " | ".join(parts)))
