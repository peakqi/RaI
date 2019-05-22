import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os
import pickle

XX = 0.2
SX1 = .5
SX2 = 0.7
RX = 0.2
LearningRate = 0.001



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
    sz_we=we.shape
    we_flat=we.reshape([sz_we[0]*sz_we[1]])
    posi_sum=sum(x for x in we_flat if x > 0)
    posi_num=np.sum(we_flat>0)
    posi_avg=posi_sum/posi_num
    neg_sum = sum(x for x in we_flat if x < 0)
    neg_num = np.sum(we_flat < 0);
    neg_avg=neg_sum/neg_num

    return np.concatenate((wed_max.reshape([1]), wed_min.reshape([1]), wed_avg.reshape([1]), wed_abs_avg.reshape([1]),
                           we_std.reshape([1]), we_avg.reshape([1]),we_absavg.reshape([1]),
                           posi_num.reshape([1]),posi_avg.reshape([1]),neg_num.reshape([1]),neg_avg.reshape([1])))


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

f, a = plt.subplots(15, N_TEST_IMG)
for i in range(N_TEST_IMG):
    a[0][i].clear()
    a[0][i].imshow(np.reshape(view_data_[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())
plt.ion()

saver = tf.train.Saver()
# type = '_n_' + str(n_encoded) + '-batch' + str(BATCH_SIZE) + '-lr' + str(LearningRate) + '-' + str(XX) + '-' + str(
#     SX1) + '-' + str(SX2) + '-' + str(RX) + '-nx' + str(nx)+'-steps' + str(nx)
#saver.restore(sess, '/Users/fengqi/Pycharm_py36/QF/4900000/_n_32-batch128-lr0.001-0.2-0.5-0.7-0.2-nx0')
rows = ['{}'.format(row) for row in ['O', 'R', 'W1', 'W9', 'a', 'd_a', 'w','#+','w+','#-','w-','|w|', 'd_w', '|d_w|', 'L']]
LearningRate = 0.001
ph_lr_ = np.ones(shape=[]) * LearningRate


nx = 0
count=0

for step in range(100001):

    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b_x, para = affine_transform1(b_x)

    _ = sess.run(train, {tf_x: b_x, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_lr: ph_lr_, ph_dis_e: ph_dis_e_})
    if step % 2000 == 0:
        loss_ = sess.run(loss, {tf_x: test_x_, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_lr: ph_lr_,
                                ph_dis_e: ph_dis_e_})
        print('step:', step, '| train loss: %.6f' % loss_)

    if step % 2000 == 0:

        view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_, ddr_ = sess.run(
            [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
             weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr, en0, en1, en2, en3, ff_encoded, de0, de1,
             de2, de3, decoded],
            {tf_x: view_data_,
             ph_encoded: view_ph_encoded_,
             ph_switch: view_ph_switch_,
             ph_dis_e: view_ph_dis_e_})

        if count == 0:
            we0p = we0;
            we1p = we1;
            we2p = we2;
            we3p = we3;
            wemp = wem;
            wd0p = wd0;
            wd1p = wd1;
            wd2p = wd2;
            wd3p = wd3;
            wddp = wdd;
            ae0p = en0_;
            ae1p = en1_;
            ae2p = en2_;
            ae3p = en3_;
            aemp = ff_encoded_;
            ad0p = de0_;
            ad1p = de1_;
            ad2p = de2_;
            ad3p = de3_;
            addp = ddr_;

        act = np.concatenate([en0_, en1_, en2_, en3_, ff_encoded_, de0_, de1_, de2_, de3_], axis=1);
        dact = np.concatenate(
            [en0_ - ae0p, en1_ - ae1p, en2_ - ae2p, en3_ - ae3p, ff_encoded_ - aemp, de0_ - ad0p, de1_ - ad1p,
             de2_ - ad2p, de3_ - ad3p], axis=1);

        sz_act = act.shape;
        sz_act_length = sz_act[1];
        col = np.ones([1, n_l0]);
        col = np.concatenate([col, np.ones([1, n_l1]) * 0.8, np.ones([1, n_l2]) * 0.6, np.ones([1, n_l3]) * 0.4,
                              np.ones([1, n_encoded]) * 0.2,
                              np.ones([1, n_d0]) * 0.4, np.ones([1, n_d1]) * 0.6, np.ones([1, n_d2]) * 0.8,
                              np.ones([1, n_d3]) * 1.0], axis=1)

        wd = np.zeros([10, 11])
        wd[0, :] = cal_w_stats(we0, we0p)  # (wed_max, wed_min, wed_avg, we_std)
        wd[1, :] = cal_w_stats(we1, we1p)
        wd[2, :] = cal_w_stats(we2, we2p)
        wd[3, :] = cal_w_stats(we3, we3p)
        wd[4, :] = cal_w_stats(wem, wemp)
        wd[5, :] = cal_w_stats(wd0, wd0p)
        wd[6, :] = cal_w_stats(wd1, wd1p)
        wd[7, :] = cal_w_stats(wd2, wd2p)
        wd[8, :] = cal_w_stats(wd3, wd3p)
        wd[9, :] = cal_w_stats(wdd, wddp)
        if count == 0:
            # wd_diff_avg = np.divide(wd[:, 2].reshape([1, 10]), wd[:, 4].reshape([1, 10]))
            # wd_diff_absavg = np.divide(wd[:, 3].reshape([1, 10]), wd[:, 4].reshape([1, 10]))
            wd_diff_avg = wd[:, 2].reshape([1, 10])
            wd_diff_absavg = wd[:, 3].reshape([1, 10])
            wd_avg = wd[:, 5].reshape([1, 10])
            wd_absavg = wd[:, 6].reshape([1, 10])
            loss_rec = loss_.reshape([1])
            wd_posi_num = wd[:,7].reshape([1,10])
            wd_posi_avg = wd[:, 8].reshape([1,10])
            wd_neg_num = wd[:, 9].reshape([1,10])
            wd_neg_avg = wd[:, 10].reshape([1,10])


        else:
            # aa = np.divide(wd[:, 2], wd[:, 4])
            # wd_diff_avg = np.concatenate((wd_diff_avg, aa.reshape([1, 10])), axis=0)
            # aa = np.divide(wd[:, 3], wd[:, 4])
            # wd_diff_absavg = np.concatenate((wd_diff_absavg, aa.reshape([1, 10])), axis=0)
            wd_diff_avg = np.concatenate((wd_diff_avg, wd[:, 2].reshape([1, 10])), axis=0)
            wd_diff_absavg = np.concatenate((wd_diff_absavg, wd[:, 3].reshape([1, 10])), axis=0)
            wd_avg = np.concatenate((wd_avg, wd[:, 5].reshape([1, 10])), axis=0)
            wd_absavg = np.concatenate((wd_absavg, wd[:, 6].reshape([1, 10])), axis=0)
            loss_rec = np.concatenate((loss_rec, loss_.reshape([1])), axis=0)
            wd_posi_num = np.concatenate((wd_posi_num, wd[:, 7].reshape([1, 10])), axis=0)
            wd_posi_avg = np.concatenate((wd_posi_avg, wd[:, 8].reshape([1, 10])), axis=0)
            wd_neg_num = np.concatenate((wd_neg_num, wd[:, 9].reshape([1, 10])), axis=0)
            wd_neg_avg = np.concatenate((wd_neg_avg, wd[:, 10].reshape([1, 10])), axis=0)


        count = count + 1


        we0p = we0;
        we1p = we1;
        we2p = we2;
        we3p = we3;
        wemp = wem;
        wd0p = wd0;
        wd1p = wd1;
        wd2p = wd2;
        wd3p = wd3;
        wddp = wdd;
        ae0p = en0_;
        ae1p = en1_;
        ae2p = en2_;
        ae3p = en3_;
        aemp = ff_encoded_;
        ad0p = de0_;
        ad1p = de1_;
        ad2p = de2_;
        ad3p = de3_;
        addp = ddr_;

    if step==100000  :
        checkstr= '/Users/fengqi/Pycharm_py36/QF/'+str(step) +'/' +'_n_' + str(n_encoded) + '-batch' + str(BATCH_SIZE) + '-lr' + str(LearningRate) + '-' + str(
            XX) + '-' + str(SX1) + '-' + str(SX2) + '-' + str(RX) + '-nx' + str(nx)
        saver.save(sess,checkstr)
        for i in range(N_TEST_IMG):
            print('step:', step,
                  ';  i:', i,
                  ';  w: %.6f' % wd[i, 5],
                  '; |w|: %.6f' % wd[i, 6],
                  ';w+_num: ', wd[i, 7],
                  ';w+_avg: %.6f' % wd[i, 8],
                  ';w-_num: ', wd[i, 9],
                  ';w-_avg: %.6f' % wd[i, 10],
                  ';|d_w|: %.6f' % wd[i, 3],
                  ';d_w: %.6f' % wd[i, 2])

            # a[1][i].set_title(str('%.2f' % np.round(wd[i, 2] * 10000, decimals=2)))
            #recon
            a[1][i].imshow(np.reshape(view_decoded_data[i], (28, 28)), cmap='gray')

            #W1
            a[2][i].clear()
            a[2][i].imshow(np.reshape(we0[:, i], (28, 28)), cmap='gray')
            # W_end
            a[3][i].imshow(np.reshape(wdd[i], (28, 28)), cmap='gray')

            # action
            a[4][i].scatter(np.arange(sz_act_length), act[i], np.ones(sz_act_length) *0.1,col.reshape([sz_act_length]))
            a[4][i].set_ylim((0, 1))

            # delta_action
            a[5][i].clear()
            a[5][i].scatter(np.arange(sz_act_length), dact[i], np.ones(sz_act_length) *0.1,col.reshape([sz_act_length]))
            a[5][i].set_ylim((-1, 1))

            # averaged W
            a[6][i].scatter(np.arange(count), wd_avg[:, i], np.ones(count) *0.1, np.ones(count))

            #posi neuron
            a[7][i].scatter(np.arange(count), wd_posi_num[:, i], np.ones(count) *0.1, np.ones(count))
            a[8][i].scatter(np.arange(count), wd_posi_avg[:, i], np.ones(count) *0.1, np.ones(count))

            #neg neuron
            a[9][i].scatter(np.arange(count), wd_neg_num[:, i], np.ones(count) *0.1, np.ones(count))
            a[10][i].scatter(np.arange(count), wd_neg_avg[:, i], np.ones(count) *0.1, np.ones(count))

            # averaged abs W
            a[11][i].scatter(np.arange(count), wd_absavg[:, i], np.ones(count) *0.1, np.ones(count))
            # averaged W(t)-W(t-1)
            a[12][i].scatter(np.arange(count), wd_diff_avg[:, i], np.ones(count) *0.1, np.ones(count))
            # averaged |W(t)-W(t-1)|
            a[13][i].scatter(np.arange(count), wd_diff_absavg[:, i], np.ones(count) *0.1, np.ones(count))



            a[0][i].set_xticks(());
            a[1][i].set_xticks(());
            a[2][i].set_xticks(());
            a[3][i].set_xticks(());
            a[4][i].set_xticks(());
            a[5][i].set_xticks(());
            a[6][i].set_xticks(());
            a[7][i].set_xticks(());
            a[8][i].set_xticks(());
            a[9][i].set_xticks(());
            a[10][i].set_xticks(());
            a[11][i].set_xticks(());
            a[12][i].set_xticks(());
            a[13][i].set_xticks(());
            a[14][i].set_xticks(());

            a[0][i].set_yticks(());
            a[1][i].set_yticks(());
            a[2][i].set_yticks(());
            a[3][i].set_yticks(());
            a[4][i].set_yticks(());
            a[5][i].set_yticks(());
            a[6][i].set_yticks(());
            a[7][i].set_yticks(());
            a[8][i].set_yticks(());
            a[9][i].set_yticks(());
            a[10][i].set_yticks(());
            a[11][i].set_yticks(());
            a[12][i].set_yticks(());
            a[13][i].set_yticks(());
            a[14][i].set_yticks(());

        a[14][0].scatter(np.arange(count), loss_rec, np.ones(count) *0.1, np.ones(count))
        a[14][0].set_ylim((0., 0.02))
        a[14][i].set_xticks(());

        for ax, row in zip(a[:, 0], rows):
            ax.set_ylabel(row, rotation=90, size='small')

        type = '_n_' + str(n_encoded) + '-batch' + str(BATCH_SIZE) + '-lr' + str(LearningRate) + '-' + str(
            XX) + '-' + str(SX1) + '-' + str(SX2) + '-' + str(RX) + '-nx' + str(nx) + '-steps' + str(step)
        subtitile = 'step' + str(step) + ';loss='
        plt.draw();
        f.suptitle(str(subtitile + '%.5f' % loss_));
        # f.tight_layout()
        plt.savefig('./' + type + '_d_w.png')

a = 1








# min1 = np.min(wd_avg, axis=0)
#         max1 = np.max(wd_avg, axis=0)
#
#         min2 = np.min(wd_absavg, axis=0)
#         max2 = np.max(wd_absavg, axis=0)
#
#         min3 = np.min(wd_diff_avg, axis=0)
#         max3 = np.max(wd_diff_avg, axis=0)
#
#         min4 = np.min(wd_diff_absavg, axis=0)
#         max4 = np.max(wd_diff_absavg, axis=0)
#
#         min5 = np.min(wd_posi_num, axis=0)
#         max5 = np.max(wd_posi_num, axis=0)
#
#         min6 = np.min(wd_posi_avg, axis=0)
#         max6 = np.max(wd_posi_avg, axis=0)
#
#
#         min7 = np.min(wd_neg_num, axis=0)
#         max7 = np.max(wd_neg_num, axis=0)
#
#         min8 = np.min(wd_neg_avg, axis=0)
#         max8 = np.max(wd_neg_avg, axis=0)