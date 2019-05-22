import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os
import scipy as sp
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

LearningRate = 0.001

# shrink to 0.60
def affine_transform_60(b_x):

    xx = np.random.uniform(low=0, high=0, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=0, high=0, size=1)
    sx = np.random.uniform(low=0.60, high=0.60, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=0.60, high=0.60, size=1)
    rr = np.random.uniform(low=0, high=0, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x, np.concatenate((xx, yy, rr), axis=0)

# mv x
def affine_transform_rand_x(b_x):
    xx = np.random.uniform(low=-0.2, high=0.2, size=1) # (-0.5,0.5)
    yy = 0
    sx = 1  # (.5,1.2)
    sy = 1
    rr = 0 # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

# mv y
def affine_transform_rand_y(b_x):
    xx = 0  # (-0.5,0.5)
    yy = np.random.uniform(low=-0.2, high=0.2, size=1)
    sx = 1  # (.5,1.2)
    sy = 1
    rr = 0 # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

# rtt
def affine_transform_rand_r(b_x):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx = 1  # (.5,1.2)
    sy = 1
    rr = np.random.uniform(low=-0.2, high=0.2, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

def affine_transform_rand_s(b_x):
    xx = 0  # (-0.5,0.5)
    yy = 0
    s=np.random.uniform(low=0.5, high=0.7, size=1)  # (-0.5,0.5)
    sx = s  # (.5,1.2)
    sy = s
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x
#scale x=deltaS
def affine_transform_delta_s(b_x,deltaS):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx = deltaS # (.5,1.2)
    sy = deltaS
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

def affine_transform_delta_r(b_x,deltaR):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx = 1 # (.5,1.2)
    sy = 1
    rr = deltaR  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

#mv x=deltaX
def affine_transform_delta_x(b_x,deltaX):
    xx = deltaX  # (-0.5,0.5)
    yy = 0
    sx = 1 # (.5,1.2)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

def affine_transform_delta_y(b_x,deltaY):
    xx = 0 # (-0.5,0.5)
    yy = deltaY
    sx = 1 # (.5,1.2)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x

XX = 0.2
SX1 = .5
SX2 = 0.7
RX = 0.2
# total mv
def affine_transform1(b_x):
    xx = np.random.uniform(low=-XX, high=XX, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-XX, high=XX, size=1)
    sx = np.random.uniform(low=SX1, high=SX2, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=SX1, high=SX2, size=1)
    rr = np.random.uniform(low=-RX, high=RX, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x, np.concatenate((xx, yy, rr), axis=0)

def cal_w_stats(we, wep):
    wed = we - wep;
    wed_max = np.amax(wed);
    wed_min = np.amin(wed);
    wed_avg = np.average(wed);
    wed_abs_avg = np.average(np.abs(wed));
    we_std = np.std(we);
    we_avg = np.average(we)
    we_absavg = np.average(np.abs(we))
    return np.concatenate((wed_max.reshape([1]), wed_min.reshape([1]), wed_avg.reshape([1]), wed_abs_avg.reshape([1]),
                           we_std.reshape([1]), we_avg.reshape([1]),we_absavg.reshape([1])))


tf.set_random_seed(1)
BATCH_SIZE = 128
N_TEST_IMG = 10
x_step=100;x_step2=x_step/2;
x=(np.arange(x_step)-x_step2)/x_step2*0.2
mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
test_x = mnist.test.images[:BATCH_SIZE]
test_x, para = affine_transform_60(test_x) #(128, 784) All chg to 0.60 size
test_x_=np.zeros([x_step,BATCH_SIZE,784])
for i in range(x_step):
    test_x_[i,:,:]=affine_transform_delta_r(test_x,x[i]) #mv x=deltaX (deltx,Batch,784)



v_step=10;v_step2=v_step/2;VIEW_SIZE=3
v=(np.arange(v_step)-v_step2)/v_step2*0.2
view_x = mnist.test.images[BATCH_SIZE+1:BATCH_SIZE+1+VIEW_SIZE]
view_x, para = affine_transform_60(view_x)

view_x_=np.zeros([v_step,VIEW_SIZE,784])
for i in range(v_step):
    view_x_[i,:,:]=affine_transform_delta_r(view_x,v[i]) #mv x=deltaX (deltx,Batch,784)


scale1 = 4
n_l0 = 64 * scale1;
n_l1 = 32 * scale1;
n_l2 = 16 * scale1;
n_l3 = 8 * scale1;
n_encoded = 32  # 4*scale1#pow(4,ii)
n_d0 = 8 * scale1;
n_d1 = 16 * scale1;
n_d2 = 32 * scale1;
n_d3 = 64 * scale1;
n_decoded = 784

print(n_encoded)
type = '_n_' + str(n_encoded) + '-batch' + str(BATCH_SIZE) + '-lr' + str(LearningRate) + '-' + str(XX) + '-' + str(
    SX1) + '-' + str(SX2) + '-' + str(RX) + '-'

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 28 * 28])  # value in the range of (0, 1)
ph_encoded = tf.placeholder(tf.float32, [None, n_encoded])
ph_switch = tf.placeholder(tf.float32, [1])
ph_lr = tf.placeholder(tf.float32, [])
ph_dis_e = tf.placeholder(tf.float32, [None, n_encoded])
# encoder


en0 = tf.layers.dense(tf_x, n_l0, tf.nn.sigmoid)
en1 = tf.layers.dense(en0, n_l1, tf.nn.sigmoid)
en2 = tf.layers.dense(en1, n_l2, tf.nn.sigmoid)
en3 = tf.layers.dense(en2, n_l3, tf.nn.sigmoid)
ff_encoded = tf.layers.dense(en3, n_encoded, tf.nn.sigmoid)
enc = ff_encoded * ph_switch + ph_encoded * (1 - ph_switch)
encoded = tf.multiply(enc, ph_dis_e)
# decoder
de0 = tf.layers.dense(encoded, n_d0, tf.nn.sigmoid)
de1 = tf.layers.dense(de0, n_d1, tf.nn.sigmoid)
de2 = tf.layers.dense(de1, n_d2, tf.nn.sigmoid)
de3 = tf.layers.dense(de2, n_d3, tf.nn.sigmoid)
decoded = tf.layers.dense(de3, n_decoded, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(ph_lr).minimize(loss)

weights_en0 = tf.get_default_graph().get_tensor_by_name(os.path.split(en0.name)[0] + '/kernel:0')
weights_en1 = tf.get_default_graph().get_tensor_by_name(os.path.split(en1.name)[0] + '/kernel:0')
weights_en2 = tf.get_default_graph().get_tensor_by_name(os.path.split(en2.name)[0] + '/kernel:0')
weights_en3 = tf.get_default_graph().get_tensor_by_name(os.path.split(en3.name)[0] + '/kernel:0')
weights_mid = tf.get_default_graph().get_tensor_by_name(os.path.split(ff_encoded.name)[0] + '/kernel:0')
weights_de0 = tf.get_default_graph().get_tensor_by_name(os.path.split(de0.name)[0] + '/kernel:0')
weights_de1 = tf.get_default_graph().get_tensor_by_name(os.path.split(de1.name)[0] + '/kernel:0')
weights_de2 = tf.get_default_graph().get_tensor_by_name(os.path.split(de2.name)[0] + '/kernel:0')
weights_de3 = tf.get_default_graph().get_tensor_by_name(os.path.split(de3.name)[0] + '/kernel:0')
weights_ddr = tf.get_default_graph().get_tensor_by_name(os.path.split(decoded.name)[0] + '/kernel:0')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded])
ph_switch_ = np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[BATCH_SIZE, n_encoded])






saver = tf.train.Saver()
nx = 0
type = '_n_32-batch128-lr0.001-0.2-0.5-0.7-0.2-nx0'
saver.restore(sess, '/Users/fengqi/Pycharm_py36/QF/4900000/' + type)
rows = ['{}'.format(row) for row in ['x1', 'x1_', 'a', 'a1_', 'd_a', '|d_a|','e_chg-e']]
LearningRate = 0.001
ph_lr_ = np.ones(shape=[]) * LearningRate


col = np.ones([1, n_l0]);
col = np.concatenate([col, np.ones([1, n_l1]) * 0.8, np.ones([1, n_l2]) * 0.6, np.ones([1, n_l3]) * 0.4,
                      np.ones([1, n_encoded]) * 0.2,
                      np.ones([1, n_d0]) * 0.4, np.ones([1, n_d1]) * 0.6, np.ones([1, n_d2]) * 0.8,
                      np.ones([1, n_d3]) * 1.0], axis=1)

####### compute corr #######
#test_x_ (100,128,784)
count = 0
cc=np.zeros([BATCH_SIZE,992])
pp=np.zeros([BATCH_SIZE,992])
for i in range(x_step):
    xdata=np.reshape(test_x_[i,:,:],(BATCH_SIZE,784))
    view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
        [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
         weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
         de2, de3, decoded],
        {tf_x: xdata,
         ph_encoded: ph_encoded_,
         ph_switch: ph_switch_,
         ph_dis_e: ph_dis_e_})
    act = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1)
    if i ==0:
        sz_act = act.shape; sz_act_length = sz_act[1];
        act_mat=np.zeros([x_step,BATCH_SIZE,sz_act_length]) #(100, 128, 992)

    act_mat[i,:,:] = act
for i in range(BATCH_SIZE):
    print(str(i))
    for j in range(992):
        #corrcoeff=np.corrcoef(act_mat[:,i,j],x)
        [corrcoeff,pval]=sp.stats.pearsonr(act_mat[:,i,j],x)
        cc[i,j]=corrcoeff#(128, 992)
        pp[i,j]=pval





#separated layer
start1=[0,256,384,448,480,512,544,608,736,992]
for k in range(9):
    print('layer'+str(k))
    f1, a1 = plt.subplots(4)
    a1[0].clear()
    cc_e=cc[:,start1[k]:start1[k+1]];    szc = cc_e.shape;  szc1 = szc[1]
    for j in range(BATCH_SIZE):
        a1[0].scatter(np.arange(szc1),cc_e[j],np.ones(szc1)*0.1,np.ones(szc1))
    a1[0].set_xticks(())
    a1[0].set_ylabel('cc-pt')
    a1[0].set_ylim((-1.1, 1.1))
    a1[1].scatter(np.arange(szc1),np.average(cc_e,axis=0),np.ones(szc1)*3)
    a1[1].set_ylabel('cc-avg')
    a1[1].set_xticks(())
    a1[1].set_ylim((-1.1, 1.1))

    pp_e=np.abs(pp[:,start1[k]:start1[k+1]])
    for j in range(BATCH_SIZE):
        a1[2].scatter(np.arange(szc1),pp_e[j],np.ones(szc1)*0.1,np.ones(szc1))
    a1[2].set_xticks(())
    a1[2].set_ylabel('pval-pt')
    a1[3].scatter(np.arange(szc1),np.average(pp_e,axis=0),np.ones(szc1)*3)
    a1[3].set_ylabel('pval-avg')
    a1[3].set_ylim((-0.0010, 0.01))
    f1.suptitle(str(k));
    plt.savefig('./corr_r/1layer' + str(k) + '.png')

### full layer corr
f2, a2 = plt.subplots(4)
a2[0].clear()
cc_e=cc
szc=cc_e.shape; szc1=szc[1]
col=np.reshape(col,[szc1])
for j in range(BATCH_SIZE):
    print(str(j))
    a2[0].scatter(np.arange(szc1),cc_e[j]*0+j,np.ones(szc1)*0.1,cc_e[j],cmap='bwr')
a2[0].set_xticks(())
a2[0].set_ylabel('cc-pt')
a2[1].scatter(np.arange(szc1),np.average(cc_e,axis=0),np.ones(szc1)*0.5,col,cmap='prism')
a2[1].set_ylabel('cc-avg')
a2[1].set_xticks(())
a2[1].set_ylim((-1.1, 1.1))
pp_e=pp[:,:]
for j in range(BATCH_SIZE):
    print(str(j))
    a2[2].scatter(np.arange(szc1),pp_e[j]*0+j,np.ones(szc1)*0.5,pp_e[j],cmap='gray')
a2[2].set_xticks(())
a2[2].set_ylabel('pval-pt')
a2[3].scatter(np.arange(szc1),np.average(pp_e,axis=0),np.ones(szc1)*0.5,col,cmap='prism')
a2[3].set_ylabel('pval-avg')
a2[3].set_ylim((-0.0010, 0.01))
plt.savefig('./corr_r/2whole_layer'  + '.png')



#./corr_r/test_move    significant colorful point move
cc_avg=np.average(cc_e,axis=0)
pp_avg=np.average(pp_e,axis=0)
ind=np.where(pp_avg<0.01)

col1=col*0+1
sz=col*0
ran=np.random.random([992])
for i in range(992):
    if pp_avg[i]<0.01:
        col1[i]=ran[i]
        sz[i]=0.5
    else:
        col1[i]=0
        sz[i]=0

view_ph_encoded_ = np.zeros(shape=[VIEW_SIZE, n_encoded])
view_ph_switch_ = np.ones(shape=[1])
view_ph_dis_e_ = np.ones(shape=[VIEW_SIZE, n_encoded])
f, a = plt.subplots(9, v_step)
for i in range(v_step):
    print(str(i))
    view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
        [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
         weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
         de2, de3, decoded],
        {tf_x: view_x_[i],
         ph_encoded: view_ph_encoded_,
         ph_switch: view_ph_switch_,
         ph_dis_e: view_ph_dis_e_})
    act = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1);
    enc= ff_encoded_

    a[0][i].clear()
    a[0][i].imshow(np.reshape(view_x_[i,0], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())
    a[1][i].clear()
    a[1][i].scatter(np.arange(sz_act_length),act[0],sz, col1,cmap='jet')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())
    a[1][i].set_ylim((-0.1, 1.1))
    a[2][i].clear()
    a[2][i].scatter(np.arange(32),act[0,480:512],sz[480:512], col1[480:512],cmap='jet')
    a[2][i].set_xticks(())
    a[2][i].set_yticks(())
    a[2][i].set_ylim((-0.1, 1.1))

    a[3][i].clear()
    a[3][i].imshow(np.reshape(view_x_[i,1], (28, 28)), cmap='gray')
    a[3][i].set_xticks(())
    a[3][i].set_yticks(())
    a[4][i].clear()
    a[4][i].scatter(np.arange(sz_act_length),act[1],sz, col1,cmap='jet')
    a[4][i].set_xticks(())
    a[4][i].set_yticks(())
    a[4][i].set_ylim((-0.1, 1.1))
    a[5][i].clear()
    a[5][i].scatter(np.arange(32),act[1,480:512],sz[480:512], col1[480:512],cmap='jet')
    a[5][i].set_xticks(())
    a[5][i].set_yticks(())
    a[5][i].set_ylim((-0.1, 1.1))


    a[6][i].clear()
    a[6][i].imshow(np.reshape(view_x_[i,2], (28, 28)), cmap='gray')
    a[6][i].set_xticks(())
    a[6][i].set_yticks(())
    a[7][i].clear()
    a[7][i].scatter(np.arange(sz_act_length),act[2],sz, col1,cmap='jet')
    a[7][i].set_xticks(())
    a[7][i].set_yticks(())
    a[7][i].set_ylim((-0.1, 1.1))
    a[8][i].clear()
    a[8][i].scatter(np.arange(32),act[2,480:512],sz[480:512], col1[480:512],cmap='jet')
    a[8][i].set_xticks(())
    a[8][i].set_yticks(())
    a[8][i].set_ylim((-0.1, 1.1))

plt.savefig('./corr_r/3test_move'  + '.png')

aaa=0




## disable 32_neuron


for m in range(0,32,1):
 #(3, 32)
    # enc = ff_encoded * ph_switch + ph_encoded * (1 - ph_switch)
    # encoded = tf.multiply(enc, ph_dis_e)
    print('neuron',str(m))
    f, a = plt.subplots(15, v_step)
    for i in range(v_step):
        print('vstep',str(i))
        view_ph_encoded_ = np.zeros(shape=[VIEW_SIZE, n_encoded])
        view_ph_switch_ = np.ones(shape=[1])
        view_ph_dis_e_ = np.ones(shape=[VIEW_SIZE, n_encoded])
        view_ph_dis_e_[:, m] = 1
        view_decoded_data1, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_,encoded_1, de0_, de1_, de2_, de3_, ddr_ = sess.run(
            [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
             weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded,encoded, de0, de1,
             de2, de3, decoded],
            {tf_x: view_x_[i],
             ph_encoded: view_ph_encoded_,
             ph_switch: view_ph_switch_,
             ph_dis_e: view_ph_dis_e_})
        act1 = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1);

        view_ph_encoded_ = np.zeros(shape=[VIEW_SIZE, n_encoded])
        view_ph_switch_ = np.ones(shape=[1])
        view_ph_dis_e_ = np.ones(shape=[VIEW_SIZE, n_encoded])
        view_ph_dis_e_[:, m] = 0
        view_decoded_data0, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_,encoded_0, de0_, de1_, de2_, de3_, ddr_ = sess.run(
            [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
             weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, encoded,de0, de1,
             de2, de3, decoded],
            {tf_x: view_x_[i],
             ph_encoded: view_ph_encoded_,
             ph_switch: view_ph_switch_,
             ph_dis_e: view_ph_dis_e_})
        act0 = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1);


        for l in range(3):
            a[l*5][i].clear()
            a[l*5][i].imshow(np.reshape(view_x_[i,l], (28, 28)), cmap='gray')
            a[l*5][i].set_xticks(())
            a[l*5][i].set_yticks(())
            a[l*5+1][i].clear()
            a[l*5+1][i].imshow(np.reshape(view_decoded_data1[l], (28, 28)), cmap='gray')
            a[l*5+1][i].set_xticks(())
            a[l*5+1][i].set_yticks(())

            a[l*5+2][i].clear()
            a[l*5+2][i].scatter(np.arange(32),encoded_1[l],sz[480:512], col1[480:512],cmap='jet')
            a[l*5+2][i].set_xticks(())
            a[l*5+2][i].set_yticks(())
            a[l*5+2][i].set_ylim((-0.1, 1.1))

            a[l*5+3][i].clear()
            a[l*5+3][i].imshow(np.reshape(view_decoded_data0[l], (28, 28)), cmap='gray')
            a[l*5+3][i].set_xticks(())
            a[l*5+3][i].set_yticks(())

            a[l*5+4][i].clear()
            a[l*5+4][i].scatter(np.arange(32), encoded_0[l], sz[480:512], col1[480:512], cmap='jet')
            a[l*5+4][i].set_xticks(())
            a[l*5+4][i].set_yticks(())
            a[l*5+4][i].set_ylim((-0.1, 1.1))



    # plt.show()

    plt.savefig('./corr_r/4test_disable_nx_' +str(m) + '.png')



aaa = 1



#
for k in range(32):


    f, a = plt.subplots(12, v_step)
    for i in range(v_step): #left-right
        print(str(i))

        for l in range(11): #up-down
            if l==0:
                a[0][i].clear()
                a[0][i].imshow(np.reshape(view_x_[i,0 ], (28, 28)), cmap='gray')
                a[0][i].set_xticks(())
                a[0][i].set_yticks(())

            view_ph_encoded_ = np.zeros(shape=[VIEW_SIZE, n_encoded])
            view_ph_switch_ = np.ones(shape=[1])
            view_ph_dis_e_ = np.ones(shape=[VIEW_SIZE, n_encoded])
            view_ph_dis_e_[:, k] =1- l / 10
            view_decoded_data1, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, encoded_1, de0_, de1_, de2_, de3_, ddr_ = sess.run(
                [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
                 weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded,
                 encoded, de0, de1,
                 de2, de3, decoded],
                {tf_x: view_x_[i],
                 ph_encoded: view_ph_encoded_,
                 ph_switch: view_ph_switch_,
                 ph_dis_e: view_ph_dis_e_})
            a[l+1][i].clear()
            a[l+1][i].imshow(np.reshape(view_decoded_data1[0], (28, 28)), cmap='gray')
            a[l+1][i].set_xticks(())
            a[l+1][i].set_yticks(())
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0, top=1, wspace=0.05, hspace=0.05)
    plt.savefig('./corr_r/5test_chg_nx_' + str(k) + '.png')








aaaa = 5



pp_avg = np.average(pp, axis=0)
pp_enc = pp_avg[480:512]
ind_pp = np.argsort(pp_enc) + 480

BATCH_SIZE = 1024
posi = np.zeros([BATCH_SIZE])
test_x = mnist.test.images[:BATCH_SIZE]
test_x,_ = affine_transform_60(test_x)  # (128, 784)
for i in range(BATCH_SIZE):
    posi[i] = np.random.random([1])  * 0.4 - 0.2
    test_x[i] = affine_transform_delta_r(np.reshape(test_x[i], [1, 784]), posi[i])  # mv x=deltaX (deltx,Batch,784)

ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded])
ph_switch_ = np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[BATCH_SIZE, n_encoded])

view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
    [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
     weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
     de2, de3, decoded],
    {tf_x: test_x,
     ph_encoded: ph_encoded_,
     ph_switch: ph_switch_,
     ph_dis_e: ph_dis_e_})
act = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1)

reg = linear_model.LinearRegression()
reg.fit(act[:, 480:512], posi)





posi2 = np.zeros([BATCH_SIZE])
pred_x = mnist.test.images[BATCH_SIZE:BATCH_SIZE * 2]
pred_x, para = affine_transform_60(pred_x)  # (128, 784)
for i in range(BATCH_SIZE):
    posi2[i] = np.random.random([1])  * 0.4 - 0.2
    pred_x[i] = affine_transform_delta_r(np.reshape(pred_x[i], [1, 784]), posi2[i])  # mv x=deltaX (deltx,Batch,784)

view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
    [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
     weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
     de2, de3, decoded],
    {tf_x: pred_x,
     ph_encoded: ph_encoded_,
     ph_switch: ph_switch_,
     ph_dis_e: ph_dis_e_})
act_pred = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1)

pred = reg.predict(act_pred[:, 480:512])

plt.close('all')
plt.scatter(posi2, pred)
plt.title(str(r2_score(posi2, pred)))
plt.xlabel('true rotation')
plt.ylabel('predicted rotation')
plt.savefig('./corr_r/6_prediction.png')

acc = np.zeros([32])
mse=np.zeros([32])
xtic = []
for jj in range(32):
    ind = ind_pp[0:jj + 1]
    # ind= np.arange(jj+1)+480
    reg = linear_model.LinearRegression()
    reg.fit(act[:, ind], posi)
    pred = reg.predict(act_pred[:, ind])
    acc[jj] = r2_score(posi2, pred)
    mse[jj] = mean_squared_error(posi2, pred)
    aa = ind[-1] - 480
    xtic.append(str(aa))

plt.close('all')
plt.scatter(np.arange(32), np.reshape(acc, [32]))
plt.xlabel('Add neuron# to Predictor')
plt.ylabel('Prediction performance')
plt.xticks(np.arange(32), xtic, rotation=90)
plt.savefig('./corr_r/7_R2.png')



plt.close('all')
plt.scatter(np.arange(32),np.reshape(mse,[32]),c='r')
plt.xlabel('Add neuron# to Predictor')
plt.ylabel('Mean square error')
plt.xticks(np.arange(32), xtic,rotation=90)
plt.ylim([0,np.max(mse)*1.2])
plt.savefig('./corr_r/7_mse.png')

np.save('r_posi2',posi2)
np.save('r_pred',pred)
np.save('r_mse',mse)
np.save('r_acc',acc)