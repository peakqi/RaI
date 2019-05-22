
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
score=np.zeros([5,100])
aa=np.load('/Users/fengqi/Pycharm_py36/QF/pred_1_chgno.npy')
score[0,:]=aa

aa=np.load('/Users/fengqi/Pycharm_py36/QF/pred_1_chgs.npy')
score[1,:]=aa

aa=np.load('/Users/fengqi/Pycharm_py36/QF/pred_1_chgr.npy')
score[2,:]=aa


aa=np.load('/Users/fengqi/Pycharm_py36/QF/pred_1_chgx.npy')
score[3,:]=aa

aa=np.load('/Users/fengqi/Pycharm_py36/QF/pred_1_chgy.npy')
score[4,:]=aa



plt.boxplot(score.transpose())
plt.savefig('_Fig3/Barplot_digit_pred.png')


