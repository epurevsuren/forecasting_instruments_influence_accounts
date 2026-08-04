"""
_test_analyst_direction.py — can GEMMA-AS-ANALYST call direction?

The encoder test (_test_direction_source.py) measured Gemma as a frozen
5120-dim vector and found nothing: NLP 50.7% / GEMMA 50.5% / both 50.1%.
That is expected — an embedding says what a post is ABOUT, and two tariff
posts sit in the same place whether the market rose or fell.

This tests the OTHER Gemma: gemma_analyst.py, which READS each post and
GENERATES a signed 1-hour impact per instrument (table gemma3_analyst_v1).
Those analyst_* columns are NOT in posts_scored, so they were absent from
the encoder test's "NLP only" baseline — Gemma-as-reasoner is unmeasured.

Three questions, on event rows only (the rows we would trade):
  1. Does sign(analyst_INST) alone beat a coin flip?   <- zero-shot direction
  2. Does a head trained on analyst_* beat NLP?
  3. Does analyst_* ADD anything on top of NLP?

Run: uv run python _test_analyst_direction.py
"""
import os, duckdb, numpy as np, pandas as pd, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
def log(m): print(m, flush=True)
DB=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','database.db')
c=duckdb.connect(DB,read_only=True); c.execute("SET memory_limit='2GB'"); c.execute("SET threads=2")
if not c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='gemma3_analyst_v1'").fetchone()[0]:
    raise SystemExit("gemma3_analyst_v1 not found — run gemma_analyst first.")
acols=[r[0] for r in c.execute("DESCRIBE gemma3_analyst_v1").fetchall() if r[0].startswith('analyst_')]
I=[a.replace('analyst_','') for a in acols]
sc=[r[0] for r in c.execute("DESCRIBE posts_scored").fetchall()]
keep=[x for x in sc if x not in ('id','text','date','platform','account','account_name','country','handle')
      and c.execute(f'SELECT typeof("{x}") FROM posts_scored LIMIT 1').fetchone()[0] in ('BIGINT','DOUBLE','INTEGER','FLOAT','HUGEINT','BOOLEAN')]
imp=[i for i in I if f'{i}_Impact' in {r[0] for r in c.execute("DESCRIBE training_set_FINAL").fetchall()}]
df=c.execute(f'''SELECT {",".join(f't."{i}_Impact"' for i in imp)},{",".join(f'p."{x}"' for x in keep)},
  {",".join(f'a."{x}"' for x in acols)}
  FROM training_set_FINAL t JOIN posts_scored p USING(id)
  JOIN gemma3_analyst_v1 a ON a.platform_id = t.platform||'_'||CAST(t.id AS VARCHAR)
  WHERE t.sample_weight>=0.3 ORDER BY t.date''').df()
c.close()
log(f"rows with analyst output: {len(df)}   instruments: {len(imp)}")
NLP=df[keep].apply(pd.to_numeric,errors='coerce').fillna(0.0).values.astype(np.float32)
AN =df[acols].apply(pd.to_numeric,errors='coerce').fillna(0.0).values.astype(np.float32)
N=len(df); i_tr,i_es=int(N*.70),int(N*.85)
def events(a):
    s=pd.Series(np.where(a>0,a,np.nan))
    thr=s.shift(1).rolling(2000,min_periods=500).quantile(0.90).values
    ok=(~np.isnan(thr))&(a>0); e=np.zeros(len(a),bool); e[ok]=a[ok]>thr[ok]; return e
def fit(X,y,ev):
    tr=np.where(ev[:i_tr])[0]; te=np.where(ev[i_es:])[0]+i_es
    if len(tr)<150 or len(te)<40: return np.nan,0
    m=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.04,subsample=.8,
        colsample_bytree=.6,min_child_weight=5,eval_metric='logloss',
        n_jobs=2,random_state=42).fit(X[tr],(y[tr]>0).astype(int))
    return float(((m.predict_proba(X[te])[:,1]>.5)==(y[te]>0)).mean()), len(te)
log(f"\n{'inst':<9}{'zero-shot':>11}{'NLP':>8}{'ANALYST':>9}{'both':>8}{'n':>6}{'±SE':>7}")
log("-"*58)
Z,A,B,C=[],[],[],[]
for inst in imp:
    y=df[f'{inst}_Impact'].fillna(0.0).values.astype(np.float64); ev=events(np.abs(y))
    te=np.where(ev[i_es:])[0]+i_es
    if len(te)<40: continue
    z=float((np.sign(df[f'analyst_{inst}'].values[te])==np.sign(y[te])).mean())
    a,n=fit(NLP,y,ev); b,_=fit(AN,y,ev); d,_=fit(np.hstack([NLP,AN]),y,ev)
    if not np.isfinite(a): continue
    se=0.5/np.sqrt(n)*100
    Z.append(z);A.append(a);B.append(b);C.append(d)
    log(f"{inst:<9}{z:>11.1%}{a:>8.1%}{b:>9.1%}{d:>8.1%}{n:>6}{se:>6.1f}%")
log("-"*58)
log(f"{'MEAN':<9}{np.mean(Z):>11.1%}{np.mean(A):>8.1%}{np.mean(B):>9.1%}{np.mean(C):>8.1%}")
log(f"\nzero-shot sign(analyst) beats 50% on {sum(1 for x in Z if x>0.5)}/{len(Z)}")
log(f"analyst ADDS on top of NLP on        {sum(1 for a_,c_ in zip(A,C) if c_>a_)}/{len(A)}")
log(f"\nA result only counts if it clears 50% by MORE than ~2x SE, and")
log(f"the best of {len(Z)} instruments will sit ~1.8 SE above 50% on noise alone.")
