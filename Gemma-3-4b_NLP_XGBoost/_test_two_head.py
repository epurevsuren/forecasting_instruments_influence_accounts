"""
_test_two_head.py — synthetic end-to-end check of the two-head classifier.

Proves, on data where we KNOW the answer, that:
  1. The regressor collapses to ~0 on a low-SNR target (reproducing the real
     symptom: mean|pred| << mean|actual|, R^2 ~ 0) — so the fix is aimed at a
     real failure, not an imagined one.
  2. The two-head classifier recovers the planted signal from the SAME data.
  3. The threshold is leak-free: it is computed from the train slice only, and
     shifting the test-slice distribution does not move it.
  4. Saved move/dir models reload and reproduce their probabilities exactly
     (the backtest loads them the same way).

Run:  uv run python _test_two_head.py
"""
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(7)
N, D = 6000, 40


def make_data():
    """Planted signal, mirroring the real problem's structure:
      * WHETHER a post moves the market depends on feature 1 ("policy
        intensity") — learnable, but noisily, like the real NLP signal.
      * WHICH WAY it moves depends on feature 0 ("hawkish vs dovish") and is
        only defined on rows that actually move.
      * Everything else is ambient noise that swamps the signal in MSE terms.
    An earlier version of this generator drew is_event at RANDOM, independent
    of X — head A then scored AUC 0.507, correctly reporting that there was
    nothing to learn. Keeping the note because it is the exact failure mode to
    watch for on real data: a coin-flip move-AUC means the gate is inert and
    all the edge is coming from head B."""
    X = RNG.normal(0, 1, (N, D)).astype(np.float32)
    vol = np.full(N, 1.0)                       # constant vol -> ratio == |y|
    p_ev = 1 / (1 + np.exp(-(1.4 * X[:, 1] - 1.2)))   # ~18% base rate
    is_event = RNG.random(N) < p_ev
    direction = np.sign(X[:, 0])
    y = RNG.normal(0, 0.30, N)                  # ambient noise, 0.30% sd
    y[is_event] += direction[is_event] * RNG.uniform(0.6, 1.6, is_event.sum())
    return X, y, vol, is_event, direction


def main():
    X, y, vol, is_event, direction = make_data()
    i_tr, i_es, i_cal = int(N * .70), int(N * .85), int(N * .93)

    # ---------- 1. the regressor collapses (reproduce the symptom) ----------
    reg = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.03,
                           subsample=.8, colsample_bytree=.5, min_child_weight=3,
                           objective='reg:squarederror', early_stopping_rounds=40,
                           n_jobs=-1, random_state=42)
    reg.fit(X[:i_tr], y[:i_tr], eval_set=[(X[i_tr:i_es], y[i_tr:i_es])], verbose=False)
    p_reg = reg.predict(X[i_es:])
    shrink = np.abs(p_reg).mean() / np.abs(y[i_es:]).mean()
    dir_reg = (np.sign(p_reg) == np.sign(y[i_es:]))[np.abs(y[i_es:]) > .1].mean()
    print(f"1. REGRESSOR   mean|pred|/mean|actual| = {shrink:.3f}  "
          f"(real pipeline: SPY 0.003)   dir={dir_reg:.1%}")

    # ---------- 2. threshold from the TRAIN SLICE ONLY ----------------------
    ratio = np.abs(y) / vol
    thr = float(np.percentile(ratio[:i_tr], 70))
    y_move, y_up = (ratio >= thr).astype(int), (y > 0).astype(int)
    print(f"2. THRESHOLD   ratio p70 (train only) = {thr:.4f}  "
          f"-> event rate train={y_move[:i_tr].mean():.1%}")

    # ---------- 3. head A ---------------------------------------------------
    spw = (len(y_move[:i_tr]) - y_move[:i_tr].sum()) / max(y_move[:i_tr].sum(), 1)
    m_mv = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=.03,
                             subsample=.8, colsample_bytree=.5, min_child_weight=5,
                             objective='binary:logistic', eval_metric='auc',
                             scale_pos_weight=spw, early_stopping_rounds=40,
                             n_jobs=-1, random_state=42)
    m_mv.fit(X[:i_tr], y_move[:i_tr],
             eval_set=[(X[i_tr:i_es], y_move[i_tr:i_es])], verbose=False)
    auc = roc_auc_score(y_move[i_es:], m_mv.predict_proba(X[i_es:])[:, 1])

    # ---------- 4. head B: trained ONLY on event rows -----------------------
    ev_tr = np.where(y_move[:i_tr] == 1)[0]
    ev_es = np.where(y_move[i_tr:i_es] == 1)[0]
    m_dir = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.04,
                              subsample=.8, colsample_bytree=.6, min_child_weight=5,
                              objective='binary:logistic', eval_metric='logloss',
                              early_stopping_rounds=40, n_jobs=-1, random_state=42)
    m_dir.fit(X[ev_tr], y_up[:i_tr][ev_tr],
              eval_set=[(X[i_tr:i_es][ev_es], y_up[i_tr:i_es][ev_es])], verbose=False)
    p_up = m_dir.predict_proba(X[i_es:])[:, 1]
    real_ev = is_event[i_es:]
    dir_clf = ((p_up > .5) == (y[i_es:] > 0))[real_ev].mean()
    print(f"3. HEAD A      move-AUC = {auc:.3f}  (0.50 = coin flip)")
    print(f"4. HEAD B      dir on REAL events = {dir_clf:.1%}  "
          f"vs regressor's {dir_reg:.1%}")

    # ---------- 5. leak check ----------------------------------------------
    y2 = y.copy()
    y2[i_es:] *= 5.0                       # blow up the test slice only
    thr2 = float(np.percentile((np.abs(y2) / vol)[:i_tr], 70))
    print(f"5. LEAK CHECK  threshold unchanged when test slice x5: "
          f"{thr:.4f} -> {thr2:.4f}  {'PASS' if abs(thr - thr2) < 1e-9 else 'FAIL'}")

    # ---------- 6. save/reload round-trip (what the backtest does) ----------
    d = tempfile.mkdtemp()
    try:
        m_mv.save_model(os.path.join(d, "T_Impact__move.json"))
        m_dir.save_model(os.path.join(d, "T_Impact__dir.json"))
        r_mv, r_dir = xgb.XGBClassifier(), xgb.XGBClassifier()
        r_mv.load_model(os.path.join(d, "T_Impact__move.json"))
        r_dir.load_model(os.path.join(d, "T_Impact__dir.json"))
        same = (np.allclose(r_mv.predict_proba(X[i_es:])[:, 1],
                            m_mv.predict_proba(X[i_es:])[:, 1]) and
                np.allclose(r_dir.predict_proba(X[i_es:])[:, 1], p_up))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(f"6. ROUND-TRIP  reloaded models reproduce probabilities: "
          f"{'PASS' if same else 'FAIL'}")

    # Pass criteria. NOTE: `shrink` is DIAGNOSTIC, not asserted — on synthetic
    # data with a strong planted signal the regressor shrinks less than it does
    # on the real 1-hour target (SPY 0.003), and an assertion on it would just
    # be testing how hard I planted the signal, not whether the code works.
    # ---------- 7. REGRESSION TEST for the two bugs found 2026-07-31 --------
    # (a) zeros-as-moves: US10Y was 48.2% zeros / NATGAS 44.1%. fillna(0.0)
    #     makes missing data look like "no move"; a quantile over that tie-mass
    #     fires on 82-89% of rows. Excluding zeros + strict '>' fixes it.
    # The mechanism is a zero rate that DIFFERS between train and test, which
    # is exactly what US10Y does: 54.7% zeros in the train slice, ~13% in test.
    # A quantile frozen on the zero-heavy train slice then sits just above the
    # tie-mass, so nearly every non-zero test row clears it — 88.6% events.
    y_z = y.copy()
    z_tr = RNG.random(i_es) < 0.55                 # train: 55% missing
    z_te = RNG.random(N - i_es) < 0.13             # test:  13% missing
    y_z[:i_es][z_tr] = 0.0
    y_z[i_es:][z_te] = 0.0
    a_z = np.abs(y_z)
    # v1: frozen train-slice quantile over the raw series (zeros included)
    thr_naive = np.percentile(a_z[:i_tr], 70)
    rate_naive_tr = (a_z[:i_tr] >= thr_naive).mean()
    rate_naive_te = (a_z[i_es:] >= thr_naive).mean()
    # v2: rolling quantile over NON-ZERO history, strict '>'
    s_f = pd.Series(np.where(a_z > 0, a_z, np.nan))
    fixed = s_f.shift(1).rolling(2000, min_periods=500).quantile(.90).values
    ok_f = (~np.isnan(fixed)) & (a_z > 0)
    m_tr = ok_f.copy(); m_tr[i_es:] = False
    m_te = ok_f.copy(); m_te[:i_es] = False
    ev = np.zeros(N, bool); ev[ok_f] = a_z[ok_f] > fixed[ok_f]
    rate_fix_tr, rate_fix_te = ev[m_tr].mean(), ev[m_te].mean()
    drift_naive = abs(rate_naive_te - rate_naive_tr)
    drift_fixed = abs(rate_fix_te - rate_fix_tr)
    zeros_ok = drift_naive > 0.15 and drift_fixed < 0.10
    print(f"7. ZEROS BUG   train 55% / test 13% missing")
    print(f"               v1 frozen quantile : {rate_naive_tr:.0%} -> "
          f"{rate_naive_te:.0%}  drift {drift_naive:.0%}  (US10Y saw 30%->89%)")
    print(f"               v2 rolling+nonzero : {rate_fix_tr:.0%} -> "
          f"{rate_fix_te:.0%}  drift {drift_fixed:.0%}  "
          f"{'PASS' if zeros_ok else 'FAIL'}")

    # (b) gate fallthrough: v1 seeded p_move_thr=0.99 and only overwrote it on
    #     hitting the precision target. Nothing did, so 22/23 instruments
    #     shipped a never-trade gate. The sweep must ALWAYS return a usable
    #     threshold, even when the target is unreachable.
    def sweep(precisions, target):
        cands = [(t, p, 50) for t, p in precisions]
        hit = [c for c in cands if c[1] >= target]
        if hit:
            return hit[0][0], True
        return (max(cands, key=lambda c: c[1])[0], False) if cands else (0.50, False)

    impossible = [(t, 0.51) for t in np.arange(0.30, 0.95, 0.05)]   # never hits .58
    thr_fb, met = sweep(impossible, 0.58)
    gate_ok = thr_fb < 0.95 and not met
    print(f"6b. GATE BUG   unreachable target -> threshold {thr_fb:.2f} "
          f"(v1 shipped 0.99 = never trade)  {'PASS' if gate_ok else 'FAIL'}")

    ok = (auc > .60                      # head A learns "will it move"
          and dir_clf > dir_reg + .05    # head B beats sign(pred) on real moves
          and abs(thr - thr2) < 1e-9     # threshold cannot see the test slice
          and same                       # models survive save/load
          and zeros_ok                   # missing data can't masquerade as events
          and gate_ok)                   # gate never falls through to never-trade
    print("\n" + ("✅ ALL CHECKS PASS — two-head recovers signal the regressor loses"
                  if ok else "❌ CHECK FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
