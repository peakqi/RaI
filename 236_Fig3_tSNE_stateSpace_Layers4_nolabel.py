import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os
from sklearn.manifold import TSNE
from matplotlib import cm
import scipy.misc


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




XX = 0.2
SX1 = .5
SX2 = 0.7
RX = 0.2
# total mv
def affine_transform1(b_x):
    xx = np.random.uniform(low=-XX, high=XX, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-XX, high=XX, size=1)
    sx = np.random.uniform(low=SX1, high=SX2, size=1)  # (.5,1.2)
    # sy = np.random.uniform(low=SX1, high=SX2, size=1)
    sy=sx
    rr = np.random.uniform(low=-RX, high=RX, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x, np.concatenate((xx, yy, sx,rr), axis=0)

def affine_transform_rand_noScale(b_x):
    xx = np.random.uniform(low=-0.2, high=.2, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-.2, high=.2, size=1)
    sx = np.random.uniform(low=0.8, high=1.2, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=0.8, high=1.2, size=1)
    rr = np.random.uniform(low=-1, high=1, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x



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

def plot_with_labels(lowDWeights, labels):
    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = plt.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=3)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)




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



tf.set_random_seed(1)
BATCH_SIZE = 1024
N_TEST_IMG = 10

mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
test_x = mnist.test.images[BATCH_SIZE*1:BATCH_SIZE*2]
test_l = mnist.test.labels[BATCH_SIZE*1:BATCH_SIZE*2]
para=np.zeros([BATCH_SIZE,4])
for i in range(BATCH_SIZE):
    test_x[i, :],para[i,:]= affine_transform1(np.reshape(test_x[i], [1, 784]))


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

plot_only = 1024
vec=act[:,480:512]
kk=4
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000);
low_dim_embs = tsne.fit_transform(vec)
plt.figure(num=None, figsize=(8, 8)) #24,16 full screen
plt.cla();
X, Y = low_dim_embs[:, 0], -low_dim_embs[:, 1]
iii=np.arange(plot_only)

for x, y, s,iind in zip(X[:plot_only], Y[:plot_only], test_l[:plot_only],iii):
    c = cm.rainbow(int(255 * s / 9));
    plt.text(x, y, ' ', backgroundcolor=c, fontsize=3 )
plt.xlim(X.min(), X.max());plt.ylim(Y.min(), Y.max());
plt.savefig('./tSNE/'+'nolabel_tsne'+str(kk)+'.png')



ind_sel=np.arange(480,512)
leng=np.asarray(512-480)
for jj in range(4):
    plt.cla();
    for ii in range(plot_only):
        x=X[ii];y=Y[ii];s=test_l[ii];vc=vec[ii,:]
        iinn = np.argsort(vc)
        c = cm.rainbow(int(255 * iinn[32-jj-1] / leng));
        plt.text(x, y, iinn[32-jj-1], backgroundcolor=c, fontsize=3 )
    plt.xlim(X.min(), X.max());plt.ylim(Y.min(), Y.max());
    plt.savefig('./tSNE/'+'nolabel_tsne_importNeuron_'+str(jj)+'.png')


plt.cla();
for ii in range(BATCH_SIZE):
    x = X[ii];
    y = Y[ii];
    s = test_l[ii];
    xxx = para[ii, 0]
# for x, y, s ,xxx in zip(X[:plot_only], Y[:plot_only], test_l[:plot_only],para[:,0]):
    c = cm.rainbow(int(255 * (xxx+0.2 )/ 0.4));
    plt.text(x, y, ' ', backgroundcolor=c, fontsize=3 )
    plt.xlim(X.min(), X.max());plt.ylim(Y.min(), Y.max());
plt.savefig('./tSNE/'+'nolabel_tsne_layer'+str(kk)+'_x'+'.png')

plt.cla();
for ii in range(BATCH_SIZE):
    x = X[ii];
    y = Y[ii];
    s = test_l[ii];
    xxx = para[ii, 1]
    # for x, y, s ,xxx in zip(X[:plot_only], Y[:plot_only], test_l[:plot_only],para[:,0]):
    c = cm.rainbow(int(255 * (xxx + 0.2) / 0.4));
    plt.text(x, y, ' ', backgroundcolor=c, fontsize=3 )
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
plt.savefig('./tSNE/' + 'nolabel_tsne_layer' + str(kk) + '_y' + '.png')

plt.cla();
for ii in range(BATCH_SIZE):
    x = X[ii];
    y = Y[ii];
    s = test_l[ii];
    xxx = para[ii, 2]
    # for x, y, s ,xxx in zip(X[:plot_only], Y[:plot_only], test_l[:plot_only],para[:,0]):
    c = cm.rainbow(int(255 * (xxx - 0.5) / 0.2));
    plt.text(x, y, ' ', backgroundcolor=c, fontsize=3 )
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
plt.savefig('./tSNE/' + 'nolabel_tsne_layer' + str(kk) + '_s' + '.png')

plt.cla();
for ii in range(BATCH_SIZE):
    x = X[ii];
    y = Y[ii];
    s = test_l[ii];
    xxx = para[ii, 3]
    # for x, y, s ,xxx in zip(X[:plot_only], Y[:plot_only], test_l[:plot_only],para[:,0]):
    c = cm.rainbow(int(255 * (xxx + 0.2) / 0.4));
    plt.text(x, y, ' ', backgroundcolor=c, fontsize=3 )
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
plt.savefig('./tSNE/' + 'nolabel_tsne_layer' + str(kk) + '_r' + '.png')







canv=1000

temp=low_dim_embs[:, 0:1];temp1=np.abs(temp);temp2=temp1.max()
LEN=canv/2*0.9;mid=np.int(canv/2)-14

for jj in range(10):
    print(jj)
    IM=np.zeros([canv,canv])
    for ii in range(BATCH_SIZE):
        if test_l[ii] == jj:
            im=np.zeros([canv,canv])
            xx=low_dim_embs[ii, 0]; yy = low_dim_embs[ii, 1]
            dx=np.int((xx/temp2)*LEN);dy=np.int((yy/temp2)*LEN*1.5);
            im[mid+dy:mid+dy+28,mid+dx:mid+dx+28]=np.reshape(test_x[ii,:], (28, 28))-0.5
            if np.sum(np.sum(np.multiply(IM,im)))==0:
                IM=IM+im
    scipy.misc.imsave('./tSNE/'+ 'nolabel_tsne_layer' + str(kk)+'digit'+str(jj)+'.png', IM)



IM=np.zeros([canv,canv])
for ii in range(BATCH_SIZE):
        im=np.zeros([canv,canv])
        xx=low_dim_embs[ii, 0]; yy = low_dim_embs[ii, 1]
        dx=np.int((xx/temp2)*LEN);dy=np.int((yy/temp2)*LEN*1.5);
        im[mid+dy:mid+dy+28,mid+dx:mid+dx+28]=np.reshape(test_x[ii,:], (28, 28))-0.5
        if np.sum(np.sum(np.multiply(IM,im)))==0:
            IM=IM+im
scipy.misc.imsave('./tSNE/'+ 'nolabel_tsne_layer' + str(kk)+'all_'+'tsne.png', IM)

