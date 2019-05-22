

import matplotlib.pyplot as plt
import numpy as np
a=np.zeros([4,32])
plt.close('all')
a[0,5]=1
a[0,8]=1
a[0,17]=1
a[0,21]=1
a[0,24]=1

a[1,13]=1
a[1,22]=1
a[1,31]=1

a[2,3]=1
a[2,5]=1
a[2,7]=1
a[2,11]=1
a[2,12]=1
a[2,18]=1
a[2,19]=1
a[2,20]=1
a[2,22]=1
a[2,27]=1
a[2,30]=1

a[3,24]=1
a[3,27]=1

plt.imshow(1-a,cmap='gray')
plt.savefig('/Users/fengqi/Pycharm_py36/QF/_Fig2/SigNeuron.png')