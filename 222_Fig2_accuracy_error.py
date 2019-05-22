

import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
xm=np.load('/Users/fengqi/Pycharm_py36/QF/x_mse.npy')
xa=np.load('/Users/fengqi/Pycharm_py36/QF/x_acc.npy')

ym=np.load('/Users/fengqi/Pycharm_py36/QF/y_mse.npy')
ya=np.load('/Users/fengqi/Pycharm_py36/QF/y_acc.npy')

sm=np.load('/Users/fengqi/Pycharm_py36/QF/s_mse.npy')
sa=np.load('/Users/fengqi/Pycharm_py36/QF/s_acc.npy')

rm=np.load('/Users/fengqi/Pycharm_py36/QF/r_mse.npy')
ra=np.load('/Users/fengqi/Pycharm_py36/QF/r_acc.npy')


plt.close('all')
fig, ax1 = plt.subplots()
xx=np.arange(32)
ax1.plot(xx,xa, 'g', xx,ya, 'b', xx,sa,'r', xx,ra,'y')
ax1.set_xlabel('Add neurons to predictor')
ax1.set_ylabel('R2')
ax2 = ax1.twinx()
ax2.plot(xx,xm, 'g--', xx,ym, 'b--', xx,sm,'r--', xx,rm,'y--')
ax2.set_ylabel('Mean square error')

fig.tight_layout()
plt.show()
plt.savefig('/Users/fengqi/Pycharm_py36/QF/_Fig2/accuracy_err.png')