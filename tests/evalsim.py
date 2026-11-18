"""Compare pipeline output with synthetic ground truth (used by tests and for tuning)."""
import numpy as np
from grain import pipeline, synth, rice, sugar

def match(result, truth, max_d):
    ks = [k for k in result["kernels"] if k["cls"] != "edge"]
    pairs, used = [], set()
    for t in truth:
        best, bd = None, max_d
        for k in ks:
            if k["id"] in used: continue
            d = np.hypot(k["cx"] - t["cx"], k["cy"] - t["cy"])
            if d < bd: best, bd = k, d
        if best is not None:
            used.add(best["id"]); pairs.append((t, best))
    return pairs, len(ks)

def run(commodity, preset, seed=7, settings=None):
    sc = synth.SimScene(commodity, preset, seed)
    img = sc.read()
    s = settings or (rice.RiceSettings() if commodity == "rice" else sugar.SugarSettings())
    from grain import calibration
    cal = calibration.detect(img, marker_mm=sc.marker_mm)
    res = pipeline.analyse(img, commodity, s, cal)
    summ = pipeline.summarise(res["kernels"], commodity, s)
    truth = sc.truth()
    pairs, n_det = match(res, truth, (3.3 if commodity == "rice" else 0.6) * sc.ppm)
    return sc, img, res, summ, truth, pairs, n_det

if __name__ == "__main__":
    import sys, collections
    com = sys.argv[1]; presets = sys.argv[2:]
    for p in presets:
        sc, img, res, summ, truth, pairs, n_det = run(com, p)
        conf = collections.Counter((t["kind"], k["cls"]) for t, k in pairs)
        acc = sum(v for (a, b), v in conf.items() if a == b) / max(1, len(pairs))
        print(f"== {com}/{p}: truth {len(truth)} detected {n_det} matched {len(pairs)} cal {res['calibration']} ms {res['ms']} acc {acc:.3f}")
        wrong = {f"{a}->{b}": v for (a, b), v in conf.items() if a != b}
        print("   errors:", wrong)
        if com == "rice":
            print("   ", [(r['key'], r['wt_pct'], r['status']) for r in summ['params']], summ['verdict']['code'])
        else:
            print("   measured:", summ.get('grade'), "MA", summ.get('ma_mm'), "CV", summ.get('cv_pct'), "L*", summ.get('mean_L'), "b*", summ.get('mean_b'), summ['counts'])
            tk = [{"cls": "crystal", "sieve_mm": t["B_mm"], "_vol": t["area_mm2"] ** 1.5, "L": 90, "b": 0, "review": []} for t in truth]
            ts = sugar.summarise(tk, sugar.SugarSettings())
            print("   truth   :", ts['size_group'], "MA", ts['ma_mm'], "CV", ts['cv_pct'])
            err = [k["breadth_mm"] - t["B_mm"] for t, k in pairs]
            print("   sieve-size error mm: mean %.3f sd %.3f" % (np.mean(err), np.std(err)))
