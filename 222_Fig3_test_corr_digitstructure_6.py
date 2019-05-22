import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os
import scipy as sp
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

dig=7
tf.set_random_seed(1)
BATCH_SIZE = 1024
N_TEST_IMG = 10

mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
test_x = mnist.test.images[:BATCH_SIZE]
#test_x, para = affine_transform_60(test_x) #(128, 784) All chg to 0.60 size
x=np.zeros([10,BATCH_SIZE])
for i in range(BATCH_SIZE):
    x[mnist.test.labels[i],i]=1
    test_x[i, :] = affine_transform_rand_r(np.reshape(test_x[i], [1, 784]))  # rtt
    test_x[i, :] = affine_transform_rand_s(np.reshape(test_x[i], [1, 784])) #rtt
    test_x[i, :] = affine_transform_rand_x(np.reshape(test_x[i], [1, 784])) #mv y
    test_x[i, :] = affine_transform_rand_y(np.reshape(test_x[i], [1, 784])) #mv y





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

cc=np.zeros([10,992])
pp=np.zeros([10,992])

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

for i in range(10):
    for j in range(992):
        [corrcoeff, pval] = sp.stats.pearsonr(x[i],act[:,j])
        cc[i,j]=corrcoeff#(128, 992)
        pp[i,j]=pval





#separated layer
start1=[0,256,384,448,480,512,544,608,736,992]

colrand=np.random.random([992])
for k in range(9): #layers
    print('layer'+str(k))
    cc_e = cc[:, start1[k]:start1[k + 1]];
    pp_e = pp[:, start1[k]:start1[k + 1]];
    szc = cc_e.shape;
    szc1 = szc[1]; col2=colrand[start1[k]:start1[k + 1]]
    f1, a1 = plt.subplots(10,2) # digits
    for l in range(10): # digits
        a1[l][0].clear()
        a1[l][0].scatter(np.arange(szc1), cc_e[l],np.ones(szc1)*0.2,col2,cmap='jet')
        a1[l][0].set_ylabel(str(l))
        a1[l][0].set_yticks(())
        a1[l][0].set_ylim((-1., 1))

        a1[l][1].clear()
        a1[l][1].scatter(np.arange(szc1), pp_e[l],np.ones(szc1)*0.2,col2,cmap='jet')
        a1[l][1].set_ylabel(str(l))
        a1[l][1].set_ylim((-0., 0.05))
        a1[l][1].set_yticks(())
        if l != 9:
            a1[l][1].set_xticks(())
            a1[l][0].set_xticks(())
    plt.savefig('./corr_digit/1layer' + str(k) + '.png')
    plt.close('all')
aaa=0



for k in range(9):  # layers
    print('layer' + str(k))
    cc_e = cc[:, start1[k]:start1[k + 1]];
    pp_e = pp[:, start1[k]:start1[k + 1]];
    mask=pp_e*0
    sz_pp=mask.shape
    for i in range(sz_pp[0]):
        for j in range(sz_pp[1]):
            if pp_e[i,j]<0.05:
                mask[i,j]=1


    szc = cc_e.shape;
    szc1 = szc[1];
    col2 = np.random.random([szc1])
    f1, a1 = plt.subplots(3)  # digits
    for l in range(10):  # digits
        a1[0].imshow(cc_e,cmap='bwr')
        a1[1].imshow(1-pp_e,cmap='gray')
        a1[2].imshow(mask, cmap='gray')
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('./corr_digit/2layer' + str(k) + '.png')

    plt.close('all')
aaa=1




for dig1 in range(10):
    print('digit',str(dig1))
    view_x = mnist.test.images[BATCH_SIZE+1:BATCH_SIZE+1001]
    view_l = mnist.test.labels[BATCH_SIZE+1:BATCH_SIZE+1001]
    jj = 0
    view_data=np.zeros([10,784])
    for i in range(1000):
        if dig1 == view_l[i]:
            im = view_x[i]
            im = affine_transform_rand_r(np.reshape(im, [1, 784]))  # rtt
            im = affine_transform_rand_s(np.reshape(im, [1, 784]))  # rtt
            im = affine_transform_rand_x(np.reshape(im, [1, 784]))  # mv y
            im = affine_transform_rand_y(np.reshape(im, [1, 784]))  # mv y
            view_data[jj,:]=im
            jj = jj + 1
            if jj == 10:
                break;

    pp_digi=pp[dig1]
    col1 = np.zeros([992]);
    sz = np.zeros([992])

    for i in range(992):
        if pp_digi[i] < 0.05:
            col1[i] = colrand[i]
            sz[i] = 1
        else:
            col1[i] = 0
            sz[i] = 0

    f, a = plt.subplots(10, 10)

    for i in range(10):# i row
        VIEW_SIZE=1
        a[i][0].imshow(np.reshape(view_data[i],[28,28]))
        a[i][0].set_xticks(());    a[i][0].set_yticks(())
        view_ph_encoded_ = np.zeros(shape=[VIEW_SIZE, n_encoded])
        view_ph_switch_ = np.ones(shape=[1])
        view_ph_dis_e_ = np.ones(shape=[VIEW_SIZE, n_encoded])
        view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
            [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
             weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
             de2, de3, decoded],
            {tf_x: np.reshape(view_data[i],[VIEW_SIZE,784]),
             ph_encoded: view_ph_encoded_,
             ph_switch: view_ph_switch_,
             ph_dis_e: view_ph_dis_e_})
        act = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1);

        for j in range(9):
            cc_n = cc[dig1, start1[j]:start1[j + 1]]; szc = cc_n.shape;szc1 = szc[0]
            a[i][j+1].scatter(np.arange(szc1),act[0,start1[j]:start1[j + 1]],sz[ start1[j]:start1[j + 1]],col1[ start1[j]:start1[j + 1]],cmap='jet')
            a[i][j+1].set_xticks(());a[i][j+1].set_yticks(())
            a[i][j+1].set_ylim([-0.1,1.1])
    plt.savefig('./corr_digit/3digit' + str(dig1) + '.png')

plt.close('all')


