import numpy as np, statistics as st, pandas as pd
d=[28,31,33,35,35,36,38,40,41,42,42,43,45,46,48,50,52,53,55,58,61,64,72,95]
for name,x in (("ORIGINAL (n=24)",d),("LARGEST REMOVED (n=23)",d[:-1])):
    s=pd.Series(x); q1,q3=np.percentile(x,[25,75]); iqr=q3-q1
    print(name)
    print(" mean=%.3f median=%.1f mode=%s midrange=%.1f"%(s.mean(),s.median(),st.multimode(x),(min(x)+max(x))/2))
    print(" range=%d Q1(numpy linear)=%.2f Q3=%.2f IQR=%.2f"%(max(x)-min(x),q1,q3,iqr))
    print(" sample var=%.3f sample SD=%.3f | population var=%.3f population SD=%.3f"%(np.var(x,ddof=1),np.std(x,ddof=1),np.var(x),np.std(x)))
    print(" fences: %.2f , %.2f  outliers=%s"%(q1-1.5*iqr,q3+1.5*iqr,[v for v in x if v<q1-1.5*iqr or v>q3+1.5*iqr]))
    print(" P80=%.1f P90=%.1f"%tuple(np.percentile(x,[80,90])))
