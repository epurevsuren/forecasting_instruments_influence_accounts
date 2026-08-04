import os
import duckdb, numpy as np, pandas as pd, xgboost as xgb, warnings, gc
warnings.filterwarnings('ignore')
def log(m): print(m, flush=True)
c=duckdb.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','database.db'),read_only=True)
c.execute("SET memory_limit='2GB'"); c.execute("SET threads=2")
sc=[r[0] for r in c.execute("DESCRIBE posts_scored").fetchall()]
keep=[x for x in sc if x not in ('id','text','date','platform','account','account_name','country','handle')
      and c.execute(f'SELECT typeof("{x}") FROM posts_scored LIMIT 1').fetchone()[0] in ('BIGINT','DOUBLE','INTEGER','FLOAT','HUGEINT','BOOLEAN')]
I=['SPY','QQQ','DIA','GOLD','OIL','XLE','XLF','VIX','NATGAS','COPPER','EUR_USD','USD_MXN']
df=c.execute(f'''SELECT t.id,t.platform,t.date,{",".join(f't."{i}_Impact"' for i in I)},
  {",".join(f'p."{x}"' for x in keep)} FROM training_set_FINAL t JOIN posts_scored p USING(id)
  WHERE t.sample_weight>=0.3 ORDER BY t.date''').df()
log(f"rows: {len(df)}")
kdf=pd.DataFrame({'k':(df['platform']+'_'+df['id'].astype(str)).values,'ord':np.arange(len(df))})
c.register('kdf',kdf)
log("loading embeddings...")
d=c.execute("SELECT k.ord, e.embedding FROM kdf k JOIN gemma3_embeddings_v1 e ON e.platform_id=k.k").df()
c.close()
log(f"  got {len(d)}")
rng=np.random.default_rng(0)
R=rng.normal(0,1/np.sqrt(128),(5120,128)).astype(np.float32)   # JL projection
E=np.zeros((len(df),128),np.float32); got=np.zeros(len(df),bool)
ords=d['ord'].values.astype(int)
for st in range(0,len(d),2000):
    sl=slice(st,min(st+2000,len(d)))
    M=np.vstack(d['embedding'].values[sl]).astype(np.float32)
    E[ords[sl]]=M@R; got[ords[sl]]=True; del M
del d; gc.collect()
log(f"  embeddings matched: {got.sum()}/{len(df)}")
df=df[got].reset_index(drop=True); E=E[got]
NLP=df[keep].apply(pd.to_numeric,errors='coerce').fillna(0.0).values.astype(np.float32)
N=len(df); i_tr,i_es=int(N*.70),int(N*.85)
W_,Q=2000,0.90
def events(a):
    s=pd.Series(np.where(a>0,a,np.nan))
    thr=s.shift(1).rolling(W_,min_periods=500).quantile(Q).values
    ok=(~np.isnan(thr))&(a>0); e=np.zeros(len(a),bool); e[ok]=a[ok]>thr[ok]; return e
def dirfit(X,y,ev):
    tr=np.where(ev[:i_tr])[0]; te=np.where(ev[i_es:])[0]+i_es
    if len(tr)<150 or len(te)<40: return np.nan,0
    m=xgb.XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.04,subsample=.8,
        colsample_bytree=.6,min_child_weight=5,eval_metric='logloss',
        n_jobs=2,random_state=42).fit(X[tr],(y[tr]>0).astype(int))
    p=m.predict_proba(X[te])[:,1]
    return float(((p>.5)==(y[te]>0)).mean()), len(te)
log(f"\n{'inst':<9}{'NLP only':>10}{'GEMMA only':>12}{'both':>9}{'n':>7}{'gemma adds':>12}")
log("-"*60)
A,B,C=[],[],[]
for inst in I:
    y=df[f'{inst}_Impact'].fillna(0.0).values.astype(np.float64); ev=events(np.abs(y))
    a,n=dirfit(NLP,y,ev); b,_=dirfit(E,y,ev); d2,_=dirfit(np.hstack([E,NLP]),y,ev)
    if not np.isfinite(a): continue
    A.append(a);B.append(b);C.append(d2)
    log(f"{inst:<9}{a:>10.1%}{b:>12.1%}{d2:>9.1%}{n:>7}{d2-a:>+12.1%}")
log("-"*60)
log(f"{'MEAN':<9}{np.mean(A):>10.1%}{np.mean(B):>12.1%}{np.mean(C):>9.1%}{'':>7}{np.mean(C)-np.mean(A):>+12.1%}")
log(f"\ngemma-only beats coin flip on {sum(1 for x in B if x>0.5)}/{len(B)}")
log(f"adding gemma to NLP helps on   {sum(1 for a,c_ in zip(A,C) if c_>a)}/{len(A)}")
