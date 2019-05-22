
import matplotlib.pyplot as plt
import numpy as np
plt.cla()
Ld1=np.load('/Users/fengqi/Pycharm_py36/QF/LnNew_Forgetting/loss_Ld10000.npy')
Lx1=np.load('/Users/fengqi/Pycharm_py36/QF/LnNew_Forgetting/loss_Lx10000.npy')
Ld2=np.load('/Users/fengqi/Pycharm_py36/QF/LnNew_Forgetting/loss_forget_Ld20000.npy')
Lx2=np.load('/Users/fengqi/Pycharm_py36/QF/LnNew_Forgetting/loss_forget_Lx20000.npy')
Ld=np.concatenate([Ld1,Ld2],axis=0)
Lx=np.concatenate([Lx1,Lx2],axis=0)
xx=np.arange(20002)
ind=np.arange(10001)
plt.scatter(xx[ind],Ld[ind],xx[ind]*0+0.1,c='b')
plt.scatter(xx[ind],Lx[ind],xx[ind]*0+0.1,c='r')
plt.gca().invert_yaxis()
plt.savefig('/Users/fengqi/Pycharm_py36/QF/LnNew_Forgetting/_2step_performance.png')


plt.cla()
Ld1=np.load('/Users/fengqi/Pycharm_py36/QF/LnNew_Forgetting/loss_Ld500.npy')
Lx1=np.load('/Users/fengqi/Pycharm_py36/QF/LnNew_Forgetting/loss_Lx500.npy')
sz=Lx1.shape
xx=np.arange(sz[0])
plt.scatter(xx,Ld1,xx*0+0.1,c='b')
plt.scatter(xx,Lx1,xx*0+0.1,c='r')
plt.gca().invert_yaxis()
plt.savefig('/Users/fengqi/Pycharm_py36/QF/LnNew_Forgetting/_2step_performance_inset.png')