import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os


def affine_transform2(b_x):
    b_x=b_x*0+1
    xx = np.random.uniform(low=-0., high=-0., size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-0., high=-0., size=1)
    sx = np.random.uniform(low=1/28, high=1/28, size=1)  # (.5,1.2)
    sy = np.random.uniform(low=1/2, high=1/2, size=1)
    rr = np.random.uniform(low=0, high=0, size=1)  # (-0.5,0.5)
    sz1, sz2 = b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential([iaa.Affine(translate_percent={"x": xx, "y": yy},
                                     scale={"x": sx, "y": sy}, rotate=rr * 180, )])
    images_aug = seq.augment_images(images)

    xx1 = np.random.uniform(low=-0.2, high=0.2, size=1)  # (-0.5,0.5)
    yy1 = np.random.uniform(low=-0.2, high=0.2, size=1)
    sx1 = np.random.uniform(low=1, high=4, size=1)  # (.5,1.2)
    sy1 = np.random.uniform(low=.51, high=1.2, size=1)
    rr1 = np.random.uniform(low=-1, high=1, size=1)  # (-0.5,0.5)
    seq1 = iaa.Sequential([iaa.Affine(translate_percent={"x": xx1, "y": yy1},
                                     scale={"x": sx1, "y": sy1}, rotate=rr1 * 180, )])
    images_aug1 = seq1.augment_images(images_aug)


    b_x = np.reshape(images_aug1, [sz1, sz2])
    return b_x, np.concatenate((xx, yy, sx, sy, rr), axis=0)

def cal_w_stats(we,wep):
    wed = we - wep;
    wed_max = np.amax(wed);
    wed_min = np.amin(wed);
    wed_avg = np.average(wed);
    we_std = np.std(we);
    return np.concatenate((wed_max.reshape([1]), wed_min.reshape([1]), wed_avg.reshape([1]), we_std.reshape([1])))

tf.set_random_seed(1)
BATCH_SIZE = 128
N_TEST_IMG = 10
mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
test_x = mnist.test.images[:BATCH_SIZE]
test_x_, para = affine_transform2(test_x)

view_data = mnist.test.images[:N_TEST_IMG] * 0
view_data_ = view_data * 0;
jj = 0
for i in range(100):
    if jj == mnist.test.labels[i]:
        view_data[jj, :] = mnist.test.images[i]
        view_data_[jj, :], para = affine_transform2(np.reshape(view_data[jj], [1, 784]))
        jj = jj + 1
        if jj == 10:
            break

for ii in range(1):
    n_encoded = 16  # pow(4,ii)
    print(n_encoded)
    type = 'const1full-n' + str(n_encoded) + '-'

    nx = -1

    # tf placeholder
    tf_x = tf.placeholder(tf.float32, [None, 28 * 28])  # value in the range of (0, 1)
    ph_encoded = tf.placeholder(tf.float32, [None, n_encoded])
    ph_switch = tf.placeholder(tf.float32, [1])
    ph_lr = tf.placeholder(tf.float32, [])
    ph_dis_e = tf.placeholder(tf.float32, [None, n_encoded])
    # encoder

    en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
    en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
    en2 = tf.layers.dense(en1, 32, tf.nn.tanh)
    en3 = tf.layers.dense(en2, 16, tf.nn.tanh)
    ff_encoded = tf.layers.dense(en3, n_encoded, tf.nn.tanh)
    enc = ff_encoded * ph_switch + ph_encoded * (1 - ph_switch)
    encoded = tf.multiply(enc, ph_dis_e)
    # decoder
    de0 = tf.layers.dense(encoded, 16, tf.nn.tanh)
    de1 = tf.layers.dense(de0, 32, tf.nn.tanh)
    de2 = tf.layers.dense(de1, 64, tf.nn.tanh)
    de3 = tf.layers.dense(de2, 128, tf.nn.tanh)
    decoded = tf.layers.dense(de3, 28 * 28, tf.nn.sigmoid)

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

    # saver = tf.train.Saver()
    # saver.restore(sess,'/Users/fengqi/Pycharm_py36/QF/temp')

    f, a = plt.subplots(6, N_TEST_IMG)
    for i in range(N_TEST_IMG):
        a[0][i].clear()
        a[0][i].imshow(np.reshape(view_data_[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
    plt.ion()

    ph_lr_ = np.ones(shape=[]) * 0.001
    rangemax = 50000;
    rangedvd = 100
    count=0
    # saver = tf.train.Saver()
    # saver.restore(sess, '/Users/fengqi/Pycharm_py36/QF/' + type)
    nx = 1 + nx;
    for step in range(rangemax * nx, rangemax * (nx + 1)):
        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        b_x, para = affine_transform2(b_x)

        _ = sess.run(train,
                     {tf_x: b_x, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_lr: ph_lr_, ph_dis_e: ph_dis_e_})
        if step % 100 == 0:
            loss_ = sess.run(loss, {tf_x: test_x_, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_lr: ph_lr_,
                                    ph_dis_e: ph_dis_e_})
            print('step:', step, '| train loss: %.4f' % loss_)

        if step % rangedvd == 0:
            view_decoded_data, we0, we1, we2, we3, wem, wd0, wd1, wd2, wd3, wdd, = sess.run(
                [decoded, weights_en0, weights_en1, weights_en2, weights_en3, weights_mid,
                 weights_de0, weights_de1, weights_de2, weights_de3, weights_ddr],
                {tf_x: view_data_,
                 ph_encoded: view_ph_encoded_,
                 ph_switch: view_ph_switch_,
                 ph_dis_e: view_ph_dis_e_})
            if step==0:
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
            wd=np.zeros([10,4])
            wd[0, :] = cal_w_stats(we0, we0p) #(wed_max, wed_min, wed_avg, we_std)
            wd[1, :] = cal_w_stats(we1, we1p)
            wd[2, :] = cal_w_stats(we2, we2p)
            wd[3, :] = cal_w_stats(we3, we3p)
            wd[4, :] = cal_w_stats(wem, wemp)
            wd[5, :] = cal_w_stats(wd0, wd0p)
            wd[6, :] = cal_w_stats(wd1, wd1p)
            wd[7, :] = cal_w_stats(wd2, wd2p)
            wd[8, :] = cal_w_stats(wd3, wd3p)
            wd[9, :] = cal_w_stats(wdd, wddp)






            if count==0:
                wd_avg=wd[:, 2].reshape([1,10])
                wd_avg_std=np.divide(wd[:, 2].reshape([1,10]),wd[:, 3].reshape([1,10]))
            else:
                wd_avg=np.concatenate((wd_avg,wd[:, 2].reshape([1,10])),axis=0)
                aa=np.divide(wd[:, 2], wd[:, 3])
                wd_avg_std=np.concatenate((wd_avg_std,aa.reshape([1,10])),axis=0)
            count=count+1

            for i in range(N_TEST_IMG):
                a[1][i].imshow(np.reshape(view_decoded_data[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())

                a[2][i].clear()
                a[2][i].imshow(np.reshape(we0[:, i], (28, 28)), cmap='gray')
                a[2][i].set_xticks(())
                a[2][i].set_yticks(())
                a[2][i].set_title(str('%.2f' % np.round(wd[i, 2]*10000, decimals=2))); #(wed_max, wed_min, wed_avg, we_std)
                a[3][i].imshow(np.reshape(wdd[ i], (28, 28)), cmap='gray')
                a[3][i].set_xticks(())
                a[3][i].set_yticks(())
                a[3][i].set_title(str('%.2f' % np.round(wd[i, 3], decimals=2)));
                max1=np.max(np.abs(wd_avg),axis=0)
                max2=np.max(np.abs(wd_avg_std),axis=0)
                a[4][i].scatter(np.arange(count),wd_avg[:,i],np.ones(count)*0.2)
                a[4][i].set_title(str( '%.2f' % np.round(wd[i,2]/wd[i,3]*10000,decimals=2)));
                a[4][i].set_xticks(())
                a[4][i].set_yticks(())
                a[4][i].set_ylim((-max1[i], max1[i]))

                a[5][i].scatter(np.arange(count), wd_avg_std[:, i], np.ones(count) * 0.2)
                # a[5][i].set_title(str('%.2f' % np.round(wd[i, 2] / wd[i, 3] * 10000, decimals=2)));
                a[5][i].set_xticks(())
                a[5][i].set_yticks(())
                a[5][i].set_ylim((-max2[i], max2[i]))
                # a[5][i].set_ylim((-0.5, 0.5))


            plt.draw();
            #plt.title(str(type + '%.4f' % loss_));
            plt.savefig('./' + type + str(step) + '.png')

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


    saver = tf.train.Saver()
    saver.save(sess, '/Users/fengqi/Pycharm_py36/QF/' + type)

a = 1