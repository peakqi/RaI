import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os
LearningRate = 0.001


XX1_ = -0.1;
YY1_ = 0;
SX1_ = .65;
SY1_ = .65
RX1_ = 0
subtitile = 'XX'+str(XX1_)+'--YY'+str(YY1_)+'--SX'+str(SX1_)+'--SY'+str(SY1_)+'--RX'+str(RX1_)

def affine_transform2(b_x):

    xx = np.random.uniform(low=XX1_, high=XX1_, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=YY1_, high=YY1_, size=1)
    sx = np.random.uniform(low=SX1_, high=SX1_, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=SY1_, high=SY1_, size=1)
    rr = np.random.uniform(low=-RX1_, high=RX1_, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x, np.concatenate((xx, yy, rr), axis=0)

XX0_ = 0;
YY0_ = 0;
SX0_ = .65;
SY0_ = .65
RX0_ = 0
def affine_transform3(b_x):

    xx = np.random.uniform(low=XX0_, high=XX0_, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=YY0_, high=YY0_, size=1)
    sx = np.random.uniform(low=SX0_, high=SX0_, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=SY0_, high=SY0_, size=1)
    rr = np.random.uniform(low=-RX0_, high=RX0_, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential(
        [iaa.Affine(translate_percent={"x": xx, "y": yy}, scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)
    b_x = np.reshape(images_aug, [sz1, sz2])
    return b_x, np.concatenate((xx, yy, rr), axis=0)

XX = 0.2
SX1 = .5
SX2 = 0.7
RX = 0.2
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
mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
test_x = mnist.test.images[:BATCH_SIZE]
test_x_, para = affine_transform1(test_x)

view_data = mnist.test.images[:N_TEST_IMG] * 0
view_data_ = view_data * 0;
jj = 0
for i in range(100):
    if jj == mnist.test.labels[i]:
        view_data[jj, :] = mnist.test.images[i]
        view_data_[jj, :], para = affine_transform1(np.reshape(view_data[jj], [1, 784]))
        jj = jj + 1
        if jj == 10:
            i = 100

x1_data = mnist.test.images[:N_TEST_IMG] * 0
x1_data_ = view_data * 0;
jj = 0
for i in range(100):
    if jj == mnist.test.labels[i]:
        x1_data[jj, :] = mnist.test.images[i]
        x1_data_[jj, :], para = affine_transform2(np.reshape(view_data[jj], [1, 784]))
        x1_data[jj, :], para = affine_transform3(np.reshape(view_data[jj], [1, 784]))
        jj = jj + 1
        if jj == 10:
            i = 100


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
view_ph_encoded_ = np.zeros(shape=[N_TEST_IMG, n_encoded])
view_ph_switch_ = np.ones(shape=[1])
view_ph_dis_e_ = np.ones(shape=[N_TEST_IMG, n_encoded])



saver = tf.train.Saver()
nx = 0
type = '_n_' + str(n_encoded) + '-batch' + str(BATCH_SIZE) + '-lr' + str(LearningRate) + '-' + str(XX) + '-' + str(
     SX1) + '-' + str(SX2) + '-' + str(RX) + '-nx' + str(nx)
saver.restore(sess, '/Users/fengqi/Pycharm_py36/QF/1000000/' + type)
rows = ['{}'.format(row) for row in ['x1', 'x1_', 'a', 'a1_', 'd_a', '|d_a|','e_chg-e']]
LearningRate = 0.001
ph_lr_ = np.ones(shape=[]) * LearningRate
rangemax = 5000000;
rangedvd = 2000;rangedvd100 = 100000
count = 0

col = np.ones([1, n_l0]);
col = np.concatenate([col, np.ones([1, n_l1]) * 0.8, np.ones([1, n_l2]) * 0.6, np.ones([1, n_l3]) * 0.4,
                      np.ones([1, n_encoded]) * 0.2,
                      np.ones([1, n_d0]) * 0.4, np.ones([1, n_d1]) * 0.6, np.ones([1, n_d2]) * 0.8,
                      np.ones([1, n_d3]) * 1.0], axis=1)




################ test ############
f, a = plt.subplots(7, N_TEST_IMG)
for i in range(N_TEST_IMG):
    a[0][i].clear()
    a[0][i].imshow(np.reshape(x1_data[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())
    a[1][i].clear()
    a[1][i].imshow(np.reshape(x1_data_[i], (28, 28)), cmap='gray')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())
plt.ion()

view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
    [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
     weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
     de2, de3, decoded],
    {tf_x: x1_data,
     ph_encoded: view_ph_encoded_,
     ph_switch: view_ph_switch_,
     ph_dis_e: view_ph_dis_e_})
act = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1);
enc= ff_encoded_

view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
    [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
     weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
     de2, de3, decoded],
    {tf_x: x1_data_,
     ph_encoded: view_ph_encoded_,
     ph_switch: view_ph_switch_,
     ph_dis_e: view_ph_dis_e_})
act_ = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1);
enc_= ff_encoded_
sz_act = act.shape;
sz_act_length = sz_act[1];
sz_enc = enc.shape;
sz_enc_length=sz_enc[1];


col_enc=np.random.random(sz_enc_length)
for i in range(N_TEST_IMG):

    # action
    a[2][i].clear()
    a[2][i].scatter(np.arange(sz_act_length), act[i], np.ones(sz_act_length) * 0.1,col.reshape([sz_act_length]))
    a[2][i].set_xticks(())
    a[2][i].set_yticks(())
    a[2][i].set_ylim((0, 1))
    # action_
    a[3][i].clear()
    a[3][i].scatter(np.arange(sz_act_length), act_[i], np.ones(sz_act_length) * 0.1,col.reshape([sz_act_length]))
    a[3][i].set_xticks(())
    a[3][i].set_yticks(())
    a[3][i].set_ylim((0, 1))

    # diff_action
    a[4][i].clear()
    a[4][i].scatter(np.arange(sz_act_length), act_[i]-act[i], np.ones(sz_act_length) * 0.1,col.reshape([sz_act_length]))
    a[4][i].set_xticks(())
    a[4][i].set_yticks(())
    a[4][i].set_ylim((-1.1, 1.1))

    # |diff_action|
    a[5][i].clear()
    a[5][i].scatter(np.arange(sz_act_length), np.abs(act_[i]-act[i]), np.ones(sz_act_length) * 0.1,col.reshape([sz_act_length]))
    a[5][i].set_xticks(())
    a[5][i].set_yticks(())
    a[5][i].set_ylim((0, 1.1))

    a[6][i].clear()
    a[6][i].scatter(np.arange(sz_enc_length), enc_[i]-enc[i], np.ones(sz_enc_length) * 0.1, col_enc)
    a[6][i].set_xticks(())
    a[6][i].set_yticks(())
    a[6][i].set_ylim((-1.1, 1.1))

    # mi=np.min(act_[i]-act[i]);ma=np.max(act_[i]-act[i]);
    #
    # print( 'layer_i:',i, '| min: %.6f' % mi,'| max: %.6f' % ma)
for ax, row in zip(a[:, 0], rows):
    ax.set_ylabel(row, rotation=90, size='small')

# type = '_n_' + str(n_encoded) + '-batch' + str(BATCH_SIZE) + '-lr' + str(LearningRate) + '-' + str(
#     XX) + '-' + str(SX1) + '-' + str(SX2) + '-' + str(RX) + '-nx' + str(nx) + '-steps' + str(step)

f.suptitle(str(subtitile));
f.tight_layout()
plt.draw();
plt.savefig('./' + subtitile + '.png')





a = 1

