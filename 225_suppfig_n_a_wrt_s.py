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
x=(np.arange(x_step)-x_step2)/x_step2*0.1+0.6
mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
test_x = mnist.test.images[:BATCH_SIZE]
#test_x, para = affine_transform_60(test_x) #(128, 784) All chg to 0.60 size
for i in range(BATCH_SIZE):
    test_x[i, :] = affine_transform_rand_r(np.reshape(test_x[i], [1, 784])) #rtt
    # test_x[i, :] = affine_transform_rand_x(np.reshape(test_x[i], [1, 784])) #mv y
    # test_x[i, :] = affine_transform_rand_y(np.reshape(test_x[i], [1, 784])) #mv y
test_x_=np.zeros([x_step,BATCH_SIZE,784])
for i in range(x_step):
    test_x_[i,:,:]=affine_transform_delta_s(test_x,x[i]) #mv x=deltaX (deltx,Batch,784)



v_step=10;v_step2=v_step/2;VIEW_SIZE=1
v=(np.arange(v_step)-v_step2)/v_step2*0.1+0.6
view_x = mnist.test.images[BATCH_SIZE+1:BATCH_SIZE+1+VIEW_SIZE]
#view_x, para = affine_transform_60(view_x)
for i in range(VIEW_SIZE):
    view_x[i, :] = affine_transform_rand_r(np.reshape(view_x[i], [1, 784])) #rtt
    # view_x[i, :] = affine_transform_rand_x(np.reshape(view_x[i], [1, 784])) #mv y
    # view_x[i, :] = affine_transform_rand_y(np.reshape(view_x[i], [1, 784]))  # mv y
view_x_=np.zeros([v_step,VIEW_SIZE,784])
for i in range(v_step):
    view_x_[i,:,:]=affine_transform_delta_s(view_x,v[i]) #mv x=deltaX (deltx,Batch,784)


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
#

cc_avg=np.average(cc,axis=0)
pp_avg=np.average(pp,axis=0)
ind=np.where(pp_avg<0.01)
nx_e = cc_avg[480:512]
ind_e = np.argsort(nx_e)

xx=np.arange(10)


s0=np.zeros([10,32])
s1=s0;s2=s0
for i in range(v_step): #left-right
    view_ph_encoded_ = np.zeros(shape=[VIEW_SIZE, n_encoded])
    view_ph_switch_ = np.ones(shape=[1])
    view_ph_dis_e_ = np.ones(shape=[VIEW_SIZE, n_encoded])
    view_decoded_data1, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, encoded_1, de0_, de1_, de2_, de3_, ddr_ = sess.run(
        [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
         weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded,
         encoded, de0, de1,
         de2, de3, decoded],
        {tf_x: view_x_[i],
         ph_encoded: view_ph_encoded_,
         ph_switch: view_ph_switch_,
         ph_dis_e: view_ph_dis_e_})

    s0[i,:]=ff_encoded_[0,:]


plt.close('all')
for i in range(32):
    plt.plot(xx,s0[:,i])
plt.savefig('./supp_fig_NActivity_forChangingxysr/s0')


aaaa = 1
