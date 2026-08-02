"""
_measure_memory_signal.py — does CROSS-POST MEMORY carry signal?

The question: instead of feeding raw embedding coordinates to XGBoost (which
absorbs ~73% of feature importance and adds ZERO incremental R2), use the
embedding only as a RETRIEVAL KEY — "what happened the last few times something
like this was posted?" — and feed those OUTCOMES as features.

CAUSALITY IS THE WHOLE GAME HERE. For a post at time T we may only look at
posts strictly before T - BUFFER_H. The buffer exists because of label twins:
two near-duplicate posts 12 minutes apart share essentially the same 1-hour
label, so a naive nearest-neighbour hands the model its own answer back and
produces a beautiful fake backtest.

Run: uv run python _measure_memory_signal.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import duckdb
import warnings
warnings.filterwarnings("ignore")

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(_HERE, "..", "database.db")

BUFFER_H = 24        # neighbours must be older than this (label-twin guard)
K = 10               # neighbours to aggregate
PROJ_DIM = 256       # random projection of the 5120-dim embedding (JL lemma)
INSTS = ["SPY", "QQQ", "DIA", "GOLD", "OIL", "VIX", "XLE", "XLF",
         "USD_MXN", "COPPER", "NATGAS", "EUR_USD"]


def log(m):
    print(m, flush=True)


def main():
    t0 = time.time()
    con = duckdb.connect(DB, read_only=True)
    con.execute("SET memory_limit='1500MB'")
    con.execute("SET threads=2")

    icols = ",".join(f't."{i}_Impact"' for i in INSTS)
    log("loading posts (sample_weight >= 0.3, chronological)...")
    base = con.execute(f"""
        SELECT t.id, t.platform, t.date, {icols}
        FROM training_set_FINAL t
        WHERE t.sample_weight >= 0.3
        ORDER BY t.date
    """).df()
    N = len(base)
    log(f"  {N} posts  {base['date'].min()} -> {base['date'].max()}")

    keys = (base["platform"] + "_" + base["id"].astype(str)).tolist()
    kdf = pd.DataFrame({"k": keys, "ord": np.arange(N)})
    con.register("kdf", kdf)

    log(f"fetching embeddings + random-projecting 5120 -> {PROJ_DIM} "
        f"(Johnson-Lindenstrauss: preserves cosine, needs no fitting so it "
        f"cannot leak)...")
    rng = np.random.default_rng(0)
    R = rng.normal(0, 1.0 / np.sqrt(PROJ_DIM), (5120, PROJ_DIM)).astype(np.float32)
    Z = np.zeros((N, PROJ_DIM), dtype=np.float32)
    got = np.zeros(N, dtype=bool)

    # ONE streaming pass. A per-chunk JOIN re-scans the whole 3.9GB embedding
    # column every time (8 chunks = 8 full scans); record batches stream a
    # single scan instead.
    d = con.execute("""
        SELECT k.ord, e.embedding
        FROM kdf k JOIN gemma3_embeddings_v1 e ON e.platform_id = k.k
    """).df()
    log(f"  joined {len(d)} embedding rows — projecting...")
    ords = d["ord"].values.astype(int)
    for st in range(0, len(d), 2000):
        sl = slice(st, min(st + 2000, len(d)))
        E = np.vstack(d["embedding"].values[sl]).astype(np.float32)
        Z[ords[sl]] = E @ R
        got[ords[sl]] = True
        del E
        log(f"  projected {min(st + 2000, len(d))}/{len(d)}")
    del d
    con.close()

    log(f"  embeddings found: {int(got.sum())}/{N}")
    base = base[got].reset_index(drop=True)
    Z = Z[got]
    N = len(base)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

    ts = pd.to_datetime(base["date"], utc=True).values.astype("datetime64[s]").astype(np.int64)
    cutoff = ts - BUFFER_H * 3600           # neighbour must be strictly older

    log(f"\ncausal kNN (k={K}, buffer={BUFFER_H}h) over {N} posts...")
    # For each row i, candidates are j < i with ts[j] <= cutoff[i]. Because the
    # frame is sorted by time, that is a PREFIX — so one searchsorted gives the
    # exact causal boundary and no future row can ever be reached.
    bound = np.searchsorted(ts, cutoff, side="right")

    feats = {f"mem_{c}": np.full(N, np.nan, dtype=np.float32) for c in
             ("dist", "age_d", "n_used")}
    per_inst = {i: {f"mem_{i}_{c}": np.full(N, np.nan, dtype=np.float32)
                    for c in ("mean", "agree", "absmean")} for i in INSTS}
    Y = {i: base[f"{i}_Impact"].fillna(0.0).values.astype(np.float32) for i in INSTS}

    B = 512
    for s in range(0, N, B):
        e = min(s + B, N)
        hi = bound[s:e]
        mx = int(hi.max())
        if mx <= K:
            continue
        sims = Z[s:e] @ Z[:mx].T                      # (b, mx)
        cols = np.arange(mx)[None, :]
        sims[cols >= hi[:, None]] = -2.0              # mask everything not causal
        idx = np.argpartition(-sims, K, axis=1)[:, :K]
        rows = np.arange(e - s)[:, None]
        sv = sims[rows, idx]
        order = np.argsort(-sv, axis=1)
        idx, sv = idx[rows, order], sv[rows, order]
        valid = sv > -1.0
        nvalid = valid.sum(1)
        for bi in range(e - s):
            gi = s + bi
            nv = int(nvalid[bi])
            if nv == 0:
                continue
            nb = idx[bi, :nv]
            feats["mem_dist"][gi] = 1.0 - float(sv[bi, 0])
            feats["mem_age_d"][gi] = (ts[gi] - ts[nb[0]]) / 86400.0
            feats["mem_n_used"][gi] = nv
            for inst in INSTS:
                yv = Y[inst][nb]
                nz = yv[yv != 0]
                per_inst[inst][f"mem_{inst}_mean"][gi] = float(yv.mean())
                per_inst[inst][f"mem_{inst}_absmean"][gi] = float(np.abs(yv).mean())
                per_inst[inst][f"mem_{inst}_agree"][gi] = (
                    float((nz > 0).mean()) if len(nz) else 0.5)
        if (s // B) % 8 == 0:
            log(f"  {e}/{N}  ({time.time() - t0:.0f}s)")

    log(f"\nkNN done in {time.time() - t0:.0f}s\n")
    log("=" * 76)
    log("  DOES MEMORY CARRY SIGNAL?  (all rows strictly causal, 24h buffered)")
    log("=" * 76)
    log(f"  {'inst':<9}{'corr(mem_absmean,|a|)':>22}{'dir acc via agree':>20}{'n':>8}")
    log("-" * 76)
    ok = ~np.isnan(feats["mem_dist"])
    ca, da = [], []
    for inst in INSTS:
        y = Y[inst]
        am = per_inst[inst][f"mem_{inst}_absmean"]
        ag = per_inst[inst][f"mem_{inst}_agree"]
        m = ok & (np.abs(y) >= 0.1) & ~np.isnan(am)
        if m.sum() < 200:
            continue
        c = float(np.corrcoef(am[m], np.abs(y[m]))[0, 1])
        # direction: does neighbour agreement predict the sign?
        strong = m & (np.abs(ag - 0.5) >= 0.2)
        d = float(((ag[strong] > 0.5) == (y[strong] > 0)).mean()) if strong.sum() > 50 else np.nan
        ca.append(c)
        if d == d:
            da.append(d)
        log(f"  {inst:<9}{c:>22.3f}{(f'{d:.1%}' if d == d else 'n/a'):>20}{int(m.sum()):>8}")
    log("-" * 76)
    log(f"  MEAN size-corr = {np.mean(ca):+.3f}   "
        f"MEAN dir-acc = {np.mean(da):.1%} (n_instruments={len(da)})")
    log(f"  reference: raw-embedding size head corr +0.218, direction 50-54%")
    log("=" * 76)
    log(f"\n  novelty: median nearest-neighbour distance = "
        f"{np.nanmedian(feats['mem_dist']):.3f}")
    log(f"  median age of nearest precedent = "
        f"{np.nanmedian(feats['mem_age_d']):.0f} days")

    out = pd.DataFrame({**feats,
                        **{k: v for d in per_inst.values() for k, v in d.items()}})
    out.insert(0, "platform_id",
               (base["platform"] + "_" + base["id"].astype(str)).values)
    out.insert(1, "date", base["date"].values)
    p = os.path.join(_HERE, "memory_features.csv")
    out.to_csv(p, index=False)
    log(f"\n  💾 {p}  ({len(out)} rows, {out.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
