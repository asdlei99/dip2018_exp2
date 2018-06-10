import os
import numpy as np
from scipy.stats import mode
gt=np.loadtxt('../data/result_100.txt',dtype=np.uint8)
lst=[x for x in os.listdir('.') if 'param'==x[:5] and 'txt' in x]

def getresraw(name):
    f=np.loadtxt(name)
    return f#.argmax(axis=1)+1
def getres(name):
    f=np.loadtxt(name)
    return f.argmax(axis=1)+1
res_lst=np.zeros_like(getresraw('parampred_0_1.txt'))
for i in lst:
    res=getresraw(i)
    acc=1.*(gt==getres(i)).sum()/len(gt)
    res_lst+=res*acc
final2=res_lst.argmax(axis=1)+1
print(1.*(gt==final2).sum()/len(gt))

res_lst=[]
for i in lst:
    res=getres(i)
    res_lst.append(res)
    print(i,1.*(gt==res).sum()/len(gt))
final=np.zeros_like(res_lst[0])
tmp=np.stack(res_lst,axis=0)
for i in range(final.shape[0]):
    a,_=mode(tmp[:,i])
    final[i]=a[0]
print(1.*(gt==final).sum()/len(gt))
with open('result.txt','w') as f:
	for i in range(len(gt)):
		f.write('%d\n'%final[i])