import json, time, random as pyrandom, math
import numpy as np
import weka.core.jvm as jvm
jvm.start(packages=False, logging_level=__import__('logging').ERROR)
from weka.core.converters import Loader
from weka.core.dataset import Instances
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
R={}
L=Loader("weka.core.converters.ArffLoader")
loan=L.load_file("wk/loan.arff");loan.class_is_last()
cg=L.load_file("wk/credit-g.arff");cg.class_is_last()
out=open('out/weka_outputs.txt','w')
def J48(opts): return Classifier(classname="weka.classifiers.trees.J48",options=opts)
def split(data,seed,pct=66):
    d=Instances.copy_instances(data);d.randomize(Random(seed))
    n=int(round(d.num_instances*pct/100));return Instances.copy_instances(d,0,n),Instances.copy_instances(d,n,d.num_instances-n)
# ---- Q1B
j=J48(["-U","-M","1"]);j.build_classifier(loan);out.write("== J48 unpruned -M 1 on 12 tuples\n"+str(j)+"\n")
j2=J48([]);j2.build_classifier(loan);out.write("== J48 default (pruned C=0.25 M=2) on 12 tuples\n"+str(j2)+"\n")
e=Evaluation(loan);e.test_model(j,loan);R['loan_unpruned_train_acc']=e.percent_correct
e=Evaluation(loan);e.test_model(j2,loan);R['loan_default_train_acc']=e.percent_correct
R['loan_default_size']=(j2.jobject.measureTreeSize(),j2.jobject.measureNumLeaves())
R['loan_unpruned_size']=(j.jobject.measureTreeSize(),j.jobject.measureNumLeaves())
e=Evaluation(loan);e.crossvalidate_model(J48(["-U","-M","1"]),loan,12,Random(1));R['loan_loocv_unpruned']=e.percent_correct
e=Evaluation(loan);e.crossvalidate_model(J48([]),loan,12,Random(1));R['loan_loocv_default']=e.percent_correct
jc=J48([]);jc.build_classifier(cg);out.write("== J48 default on credit-g (full data)\n"+str(jc)+"\n")
R['cg_default']={}
grid=[]
for C in ["0.05","0.1","0.25","0.5"]:
    for M in ["1","2","5","10","20"]:
        opts=["-C",C,"-M",M]
        c=J48(opts);c.build_classifier(cg)
        e1=Evaluation(cg);e1.test_model(c,cg)
        e2=Evaluation(cg);e2.crossvalidate_model(J48(opts),cg,10,Random(1))
        grid.append(dict(C=C,M=M,size=int(c.jobject.measureTreeSize()),leaves=int(c.jobject.measureNumLeaves()),train=e1.percent_correct,cv=e2.percent_correct))
for M in ["1","2","5","10","20"]:
    opts=["-U","-M",M];c=J48(opts);c.build_classifier(cg)
    e1=Evaluation(cg);e1.test_model(c,cg);e2=Evaluation(cg);e2.crossvalidate_model(J48(opts),cg,10,Random(1))
    grid.append(dict(C="unpruned",M=M,size=int(c.jobject.measureTreeSize()),leaves=int(c.jobject.measureNumLeaves()),train=e1.percent_correct,cv=e2.percent_correct))
R['j48_grid']=grid
# ---- Q2B NaiveBayes on loan
nb=Classifier(classname="weka.classifiers.bayes.NaiveBayes");nb.build_classifier(loan);out.write("== NaiveBayes on 12 tuples\n"+str(nb)+"\n")
x=Instances.copy_instances(loan,0,1);
from weka.core.dataset import Instance
inst=Instance.create_instance([2,0,1,0],weight=1.0) if False else None
# classify X = Low, Good, Self-employed using distribution
import jpype
from weka.core.dataset import Instance
X=Instance.create_instance(np.array([2.0,0.0,1.0,0.0]))  # Low=idx2, Good=0, Self-employed=1
X.dataset=loan
R['nb_X_dist']=[float(v) for v in nb.distribution_for_instance(X)]
R['nb_X_class_order']=[str(v) for v in loan.class_attribute.values]
# ---- Q3B JRip, PART on credit-g
for name in ["weka.classifiers.rules.JRip","weka.classifiers.rules.PART"]:
    c=Classifier(classname=name);c.build_classifier(cg);out.write(f"== {name} on credit-g (full data)\n"+str(c)+"\n")
    e=Evaluation(cg);e.crossvalidate_model(Classifier(classname=name),cg,10,Random(1));R[name.split('.')[-1]+'_cv']=e.percent_correct
    R[name.split('.')[-1]+'_nrules']=str(c).count('=>') if 'JRip' in name else None
R['J48_cv_default']=[g for g in grid if g['C']=='0.25' and g['M']=='2'][0]['cv']
# Weka Apriori CARs on 12 tuples
ap=Classifier  # placeholder
from weka.associations import Associator
a=Associator(classname="weka.associations.Apriori",options=["-N","30","-T","0","-C","0.7","-D","0.05","-U","1.0","-M","0.25","-A","-c","4"])
a.build_associations(loan);out.write("== Apriori CARs on 12 tuples (minsup 0.25, conf 0.7)\n"+str(a)+"\n")
# ---- Q4B MLP
def mlp(opts): return Classifier(classname="weka.classifiers.functions.MultilayerPerceptron",options=opts)
tr,te=split(cg,1,66)
rows=[]
def run(L_,H,N,M_="0.2"):
    c=mlp(["-L",str(L_),"-M",M_,"-N",str(N),"-H",str(H),"-S","1"])
    t0=time.time();c.build_classifier(tr);dt=time.time()-t0
    e1=Evaluation(tr);e1.test_model(c,tr);e2=Evaluation(tr);e2.test_model(c,te)
    rows.append(dict(L=L_,H=H,N=N,train=e1.percent_correct,test=e2.percent_correct,time=dt))
for Lr in [0.01,0.1,0.3,0.9,1.5]: run(Lr,"5",200)
for H in ["2","5","10","20","40"]: run(0.3,H,200)
for N in [10,50,200,500,1000]: run(0.3,"10",N)
run(0.3,"40",1000);run(0.3,"a",500)
R['mlp']=rows
# decision tree comparison on same split
c=J48([]);t0=time.time();c.build_classifier(tr);dt=time.time()-t0
e1=Evaluation(tr);e1.test_model(c,tr);e2=Evaluation(tr);e2.test_model(c,te)
R['j48_split']=dict(train=e1.percent_correct,test=e2.percent_correct,time=dt)
# ---- Q6B estimation
def smo(): return Classifier(classname="weka.classifiers.functions.SMO")
def nbc(): return Classifier(classname="weka.classifiers.bayes.NaiveBayes")
mk={"J48":lambda:J48([]),"NaiveBayes":nbc,"SMO":smo}
# holdout 5+ splits for J48
hold=[]
for s in range(1,11):
    a,b=split(cg,s,66);c=J48([]);c.build_classifier(a);e=Evaluation(a);e.test_model(c,b);hold.append(e.percent_correct)
R['holdout_j48']=hold
cv=[]
for s in range(1,6):
    e=Evaluation(cg);e.crossvalidate_model(J48([]),cg,10,Random(s));cv.append(e.percent_correct)
R['cv10_j48_seeds']=cv
# bootstrap .632 for J48
n=cg.num_instances;boot=[];rng=np.random.default_rng(5)
e=Evaluation(cg);c=J48([]);c.build_classifier(cg);e.test_model(c,cg);resub=e.percent_correct
for b in range(50):
    idx=rng.integers(0,n,n);oob=sorted(set(range(n))-set(idx.tolist()))
    trn=Instances.template_instances(cg);
    for i in idx: trn.add_instance(cg.get_instance(int(i)))
    tst=Instances.template_instances(cg)
    for i in oob: tst.add_instance(cg.get_instance(i))
    c=J48([]);c.build_classifier(trn);ev=Evaluation(trn);ev.test_model(c,tst);boot.append(ev.percent_correct)
R['boot_oob_mean']=float(np.mean(boot));R['boot_oob_sd']=float(np.std(boot,ddof=1));R['resub']=resub
R['boot632']=0.368*resub+0.632*float(np.mean(boot))
# 10x10 CV per fold paired for 3 classifiers
accs={k:[] for k in mk}
for rep in range(10):
    rnd=Random(rep+1);d=Instances.copy_instances(cg);d.randomize(rnd)
    d.stratify(10)
    for f in range(10):
        a=d.train_cv(10,f);b=d.test_cv(10,f)
        for k,fn in mk.items():
            c=fn();c.build_classifier(a);ev=Evaluation(a);ev.test_model(c,b);accs[k].append(ev.percent_correct)
R['cv_acc']={k:v for k,v in accs.items()}
# corrected resampled t-test
from scipy import stats
def corrt(x,y):
    d=np.array(x)-np.array(y);n=len(d);k=10;n2n1=1/9
    var=d.var(ddof=1)
    t=d.mean()/math.sqrt((1/n+n2n1)*var);p=2*(1-stats.t.cdf(abs(t),n-1));return float(d.mean()),float(t),float(p)
R['paired']={f"{a}_vs_{b}":corrt(accs[a],accs[b]) for a,b in [("J48","NaiveBayes"),("J48","SMO"),("NaiveBayes","SMO")]}
# ROC for NaiveBayes + J48 from 10-fold CV predictions
roc={}
for k in ["J48","NaiveBayes"]:
    e=Evaluation(cg);e.crossvalidate_model(mk[k](),cg,10,Random(1))
    ys=[];ps=[]
    for p in e.predictions:
        ys.append(1 if p.actual==1.0 else 0)  # bad=index1
        ps.append(float(p.distribution[1]))
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr,tpr,_=roc_curve(ys,ps);roc[k]=dict(auc=float(roc_auc_score(ys,ps)),fpr=fpr.tolist(),tpr=tpr.tolist())
    R[k+'_cv_acc_seed1']=e.percent_correct
R['roc']=roc
json.dump(R,open('out/weka_results.json','w'),indent=1,default=str)
out.close();jvm.stop();print("done")
