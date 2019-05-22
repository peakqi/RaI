import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os
from sklearn.manifold import TSNE
from matplotlib import cm
import scipy as sp


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
    return b_x

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

def affine_transform_rand_sx(b_x):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx =np.random.uniform(low=0.5, high=0.7, size=1)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x
def affine_transform_rand_sx_full(b_x):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx =np.random.uniform(low=0.3, high=1, size=1)
    sy = 1
    rr = 0  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x


def affine_transform_delta_sx(b_x,deltaS ):
    xx = 0  # (-0.5,0.5)
    yy = 0
    sx = deltaS # (.5,1.2)
    sy = 1
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



def square(b_x):
    b_x=b_x*0+1
    xx = 0 # (-0.5,0.5)
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


BATCH_SIZE=128
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

plt.figure(num=None, figsize=(24, 16))

tf.set_random_seed(1)
BATCH_SIZE = 1024
N_TEST_IMG = 10

mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
b_x = mnist.test.images[:BATCH_SIZE]
b_y = mnist.test.labels[:BATCH_SIZE]
test_x=np.zeros([10,784])


jj=0
for i in range(BATCH_SIZE):
    if jj==b_y[i]:
        test_x[jj, :]=b_x[i, :]
        jj=jj+1
        if jj==10:
            break;

x_step=10;x_step2=x_step/2;
x=(np.arange(x_step)-x_step2)/x_step2*0.1+0.6

test_x_=np.zeros([100,784])
for ii in range(10):
    temp=affine_transform_delta_s(test_x,x[ii])
    test_x_[ii*10:(ii+1)*10,:]=temp


ph_encoded_ = np.zeros(shape=[100, n_encoded])
ph_switch_ = np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[100, n_encoded])
view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
    [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
     weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
     de2, de3, decoded],
    {tf_x: test_x_,
     ph_encoded: ph_encoded_,
     ph_switch: ph_switch_,
     ph_dis_e: ph_dis_e_})
act = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1)


start1=[0,256,384,448,480,512,544,608,736,992]

for kk in range (9):
    print('layer'+str(kk))
    f, a = plt.subplots(1, 2)
    cc = np.zeros([100, 100])
    pp = np.zeros([100, 100])
    for ii in range(100):
        for jj in range(100):
            temp1 = act[ii, start1[kk]:start1[kk + 1]]
            temp2 = act[jj, start1[kk]:start1[kk + 1]]
            ttt=sp.stats.pearsonr(temp1,temp2)
            ttnp=np.asarray(ttt)
            cc[ii,jj]=ttnp[0]
            pp[ii, jj] = ttnp[1]

    a[0].imshow(cc)
    a[1].imshow(1-pp)
    a[1].invert_yaxis()
    a[0].invert_yaxis()
    plt.savefig('./similarity/slayer'+str(kk)+'.png')
