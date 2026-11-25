import numpy as np, json, math, warnings, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RepeatedStratifiedKFold, cross_val_predict
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_circles, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score
R={}
rng=np.random.default_rng(3)
# ---------- Q2B correlated attributes
n=3000;y=rng.integers(0,2,n)
def noisy(p): return np.where(rng.random(n)<p,1-y,y)
x1,x2,x3=noisy(.30),noisy(.35),noisy(.40)
rows=[]
for k in [0,1,2,4,8]:
    X=np.column_stack([x1,x2,x3]+[x3]*k)           # k exact copies of the weakest attribute
    skf=StratifiedKFold(10,shuffle=True,random_state=1)
    res={}
    for name,m in [("NaiveBayes",CategoricalNB(alpha=1)),("DecisionTree",DecisionTreeClassifier(criterion='entropy',min_samples_leaf=30,random_state=0))]:
        a=cross_val_score(m,X,y,cv=skf,scoring='accuracy').mean()
        p=cross_val_predict(m,X,y,cv=skf,method='predict_proba')[:,1]
        res[name]=(a,roc_auc_score(y,p))
    rows.append(dict(copies=k,corr=float(np.corrcoef(x3,x3)[0,1]) if k else None,**{f"{a}_acc":v[0] for a,v in res.items()},**{f"{a}_auc":v[1] for a,v in res.items()}))
R['corr_study']=rows
# ---------- Q4 learning-rate demo (the hand network, 300 epochs)
sig=lambda z:1/(1+np.exp(-z))
def train(lr,ep=300):
    w=dict(w1h1=.2,w2h1=.4,w1h2=-.3,w2h2=.1,wh1o=-.3,wh2o=-.2,bh1=-.4,bh2=.2,bo=.1);x1,x2,t=1,0,1;err=[]
    for e in range(ep):
        o1=sig(w['w1h1']*x1+w['w2h1']*x2+w['bh1']);o2=sig(w['w1h2']*x1+w['w2h2']*x2+w['bh2']);oo=sig(w['wh1o']*o1+w['wh2o']*o2+w['bo'])
        dO=oo*(1-oo)*(t-oo);d1=o1*(1-o1)*dO*w['wh1o'];d2=o2*(1-o2)*dO*w['wh2o']
        w['wh1o']+=lr*dO*o1;w['wh2o']+=lr*dO*o2;w['bo']+=lr*dO;w['w1h1']+=lr*d1*x1;w['bh1']+=lr*d1;w['w1h2']+=lr*d2*x1;w['bh2']+=lr*d2
        err.append(t-oo)
    return err
R['lr_demo']={str(lr):[train(lr,300)[i] for i in (0,9,99,299)] for lr in [0.01,0.1,0.9,5]}
# same weights equal -> symmetry
def sym(ep=200,lr=.9):
    w1=w2=.5;b=0.1;vo=.5;x1,x2,t=1,0,1;res=[]
    for e in range(ep):
        o1=sig(.5*x1+.5*x2+b);o2=sig(.5*x1+.5*x2+b);oo=sig(vo*o1+vo*o2+.5)
        dO=oo*(1-oo)*(t-oo);d1=o1*(1-o1)*dO*vo;d2=o2*(1-o2)*dO*vo
    return o1,o2,d1,d2
R['symmetry']=sym(1)
# ---------- Q5B SVM
X,yc=make_circles(600,noise=.12,factor=.45,random_state=2)
Xtr,Xte,ytr,yte=train_test_split(X,yc,test_size=.3,random_state=1,stratify=yc)
rows=[]
for name,kw in [("linear",dict(kernel='linear')),("poly d=2",dict(kernel='poly',degree=2,coef0=1)),("poly d=3",dict(kernel='poly',degree=3,coef0=1)),("RBF γ=1",dict(kernel='rbf',gamma=1)),("RBF γ=10",dict(kernel='rbf',gamma=10)),("RBF γ=100",dict(kernel='rbf',gamma=100)),("RBF γ=1000",dict(kernel='rbf',gamma=1000))]:
    m=SVC(C=1,**kw).fit(Xtr,ytr);rows.append(dict(kernel=name,C=1,train=m.score(Xtr,ytr),test=m.score(Xte,yte),nsv=int(m.n_support_.sum())))
for C in [0.01,0.1,1,10,1000]:
    m=SVC(kernel='rbf',gamma=1,C=C).fit(Xtr,ytr);rows.append(dict(kernel="RBF γ=1",C=C,train=m.score(Xtr,ytr),test=m.score(Xte,yte),nsv=int(m.n_support_.sum())))
R['svm_circles']=rows
bc=load_breast_cancer();Xb,yb=bc.data,bc.target
skf=StratifiedKFold(10,shuffle=True,random_state=1);sc=[]
for k in [dict(kernel='linear'),dict(kernel='rbf',gamma='scale'),dict(kernel='rbf',gamma=0.01)]:
    un=cross_val_score(SVC(C=1,**k),Xb,yb,cv=skf).mean();sca=cross_val_score(make_pipeline(StandardScaler(),SVC(C=1,**k)),Xb,yb,cv=skf).mean()
    sc.append(dict(kernel=str(k),unscaled=un,scaled=sca))
R['svm_scaling']=sc
# ---------- Q7 experiments: Pima analogue with 6% positives
p=pd.read_csv('wk/pima.csv',header=None,names=['preg','glucose','bp','skin','insulin','bmi','pedigree','age','y'])
for c in ['glucose','bp','skin','insulin','bmi']: p[c]=p[c].replace(0,np.nan)
R['pima_missing']={c:int(p[c].isna().sum()) for c in ['glucose','bp','skin','insulin','bmi']};R['pima_n']=len(p)
neg=p[p.y==0];pos=p[p.y==1].sample(32,random_state=4);d=pd.concat([neg,pos]).sample(frac=1,random_state=0).reset_index(drop=True)
R['q7_data']=dict(n=len(d),pos=int(d.y.sum()),rate=float(d.y.mean()))
Xd=d.drop(columns='y');yd=d.y.values
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
def model(kind):
    imp=SimpleImputer(strategy='median',add_indicator=True)
    if kind=='tree': return make_pipeline(imp,DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_leaf=5,random_state=0))
    if kind=='tree_deep': return make_pipeline(imp,DecisionTreeClassifier(criterion='entropy',random_state=0))
    if kind=='logreg': return make_pipeline(imp,StandardScaler(),LogisticRegression(max_iter=1000))
    if kind=='logreg_bal': return make_pipeline(imp,StandardScaler(),LogisticRegression(max_iter=1000,class_weight={0:1,1:10}))
    if kind=='tree_bal': return make_pipeline(imp,DecisionTreeClassifier(criterion='entropy',max_depth=4,min_samples_leaf=5,class_weight={0:1,1:10},random_state=0))
def cvpred(kind,reps=5,resample=False):
    out=[]
    for r in range(reps):
        skf=StratifiedKFold(10,shuffle=True,random_state=r);pr=np.zeros(len(d))
        for a,b in skf.split(Xd,yd):
            Xa,ya=Xd.iloc[a],yd[a]
            if resample:     # random oversampling of positives in TRAIN folds only to 1:1
                pi=np.where(ya==1)[0];ni=np.where(ya==0)[0];rs=np.random.default_rng(r)
                extra=rs.choice(pi,size=len(ni)-len(pi),replace=True);idx=np.concatenate([ni,pi,extra]);Xa,ya=Xa.iloc[idx],ya[idx]
            m=model(kind).fit(Xa,ya);pr[b]=m.predict_proba(Xd.iloc[b])[:,1]
        out.append(pr)
    return out
def summ(prs,thr):
    tp=fn=fp=tn=0;aucs=[]
    for pr in prs:
        yh=(pr>=thr).astype(int);c=confusion_matrix(yd,yh).ravel();tn+=c[0];fp+=c[1];fn+=c[2];tp+=c[3];aucs.append(roc_auc_score(yd,pr))
    k=len(prs);tp,fn,fp,tn=[v/k for v in (tp,fn,fp,tn)]
    return dict(thr=thr,TP=tp,FN=fn,FP=fp,TN=tn,acc=(tp+tn)/len(yd),recall=tp/(tp+fn),prec=tp/(tp+fp) if tp+fp else float('nan'),auc=float(np.mean(aucs)),cost=10*fn+fp)
exp={}
for name,kind,rs,thr in [("Baseline: deep tree, threshold 0.5","tree_deep",False,.5),("Shallow tree (d≤4), threshold 0.5","tree",False,.5),
                          ("Shallow tree + cost threshold 1/11","tree",False,1/11),("Shallow tree + oversampling, thr 0.5","tree",True,.5),
                          ("Shallow tree + class weights 10:1, thr 0.5","tree_bal",False,.5),
                          ("Logistic regression, thr 0.5","logreg",False,.5),("Logistic regression + cost threshold 1/11","logreg",False,1/11),
                          ("Logistic regression + class weights 10:1, thr 0.5","logreg_bal",False,.5)]:
    prs=cvpred(kind,5,rs);exp[name]=summ(prs,thr)
    exp[name]['_prs_key']=kind+str(rs)
R['q7_exp']=exp
# threshold sweep for shallow tree and logreg
sweep={}
for kind in ['tree','logreg']:
    prs=cvpred(kind,5,False);sweep[kind]=[summ(prs,t) for t in [0.05,1/11,0.15,0.2,0.3,0.5,0.7]]
R['q7_sweep']=sweep
# always-negative and flag-all baselines
R['q7_trivial']=dict(never=dict(cost=10*int(yd.sum()),acc=1-yd.mean()),all=dict(cost=int((yd==0).sum()),acc=yd.mean()))
json.dump(R,open('out/sk_results.json','w'),indent=1,default=float)
print(json.dumps(R['q7_exp'],indent=0,default=float)[:3000])
