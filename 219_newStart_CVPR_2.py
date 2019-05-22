
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os


XX=0.2
SX1=.5
SX2=0.7
RX=0.2
LearningRate=0.001
def affine_transform1(b_x):
    xx = np.random.uniform(low=-XX, high=XX, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-XX, high=XX, size=1)
    sx = np.random.uniform(low=SX1, high=SX2, size=1)    # (.5,1.2)
    sy = np.random.uniform(low=SX1, high=SX2, size=1)
    rr = np.random.uniform(low=-RX, high=RX, size=1)  # (-0.5,0.5)
    sz1,sz2=b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential([iaa.Affine(translate_percent={"x": xx, "y":yy},scale={"x": sx, "y": sy},rotate=rr*180,)])
    images_aug = seq.augment_images(images)
    b_x=np.reshape(images_aug,[sz1,sz2])
    return b_x,np.concatenate((xx,yy,rr),axis=0)

tf.set_random_seed(1)
BATCH_SIZE = 128
N_TEST_IMG=10
mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data
test_x = mnist.test.images[:BATCH_SIZE]
test_x_, para = affine_transform1(test_x)

view_data = mnist.test.images[:N_TEST_IMG]*0
view_data_=view_data*0;
jj=0
for i in range(100):
    if jj==mnist.test.labels[i]:
        view_data[jj,:]=mnist.test.images[i]
        view_data_[jj,:], para = affine_transform1(np.reshape(view_data[jj],[1,784]))
        jj=jj+1
        if jj==10:
            i=100


for ii in range(1):
    scale1=4
    n_l0=64*scale1;n_l1=32*scale1;n_l2=16*scale1;n_l3=8*scale1; n_encoded = 32#4*scale1#pow(4,ii)
    n_d0=8*scale1; n_d1=16*scale1;n_d2=32*scale1;n_d3=64*scale1;n_decoded = 784

    print(n_encoded)

    # tf placeholder
    tf_x = tf.placeholder(tf.float32, [None, 28*28])    # value in the range of (0, 1)
    ph_encoded = tf.placeholder(tf.float32, [None, n_encoded])
    ph_switch =  tf.placeholder(tf.float32, [1])
    ph_lr=tf.placeholder(tf.float32, [])
    ph_dis_e=tf.placeholder(tf.float32, [None,n_encoded])
    # encoder


    en0 = tf.layers.dense(tf_x, n_l0, tf.nn.sigmoid)
    en1 = tf.layers.dense(en0, n_l1, tf.nn.sigmoid)
    en2 = tf.layers.dense(en1, n_l2, tf.nn.sigmoid)
    en3 = tf.layers.dense(en2, n_l3, tf.nn.sigmoid)
    ff_encoded = tf.layers.dense(en3, n_encoded,tf.nn.sigmoid)
    enc=ff_encoded*ph_switch+ph_encoded*(1-ph_switch)
    encoded=tf.multiply(enc,ph_dis_e)
    # decoder
    de0 = tf.layers.dense(encoded, n_d0, tf.nn.sigmoid)
    de1 = tf.layers.dense(de0, n_d1, tf.nn.sigmoid)
    de2 = tf.layers.dense(de1, n_d2, tf.nn.sigmoid)
    de3 = tf.layers.dense(de2, n_d3, tf.nn.sigmoid)
    decoded = tf.layers.dense(de3, n_decoded, tf.nn.sigmoid)

    loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
    train = tf.train.AdamOptimizer(ph_lr).minimize(loss)
    weights = tf.get_default_graph().get_tensor_by_name(os.path.split(en0.name)[0] + '/kernel:0')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())






    ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded])
    ph_switch_= np.ones(shape=[1])
    ph_dis_e_ = np.ones(shape=[BATCH_SIZE, n_encoded])
    view_ph_encoded_ = np.zeros(shape=[N_TEST_IMG, n_encoded])
    view_ph_switch_ = np.ones(shape=[1])
    view_ph_dis_e_ = np.ones(shape=[N_TEST_IMG, n_encoded])




    f, a = plt.subplots(3, N_TEST_IMG)
    for i in range(N_TEST_IMG):
        a[0][i].clear()
        a[0][i].imshow(np.reshape(view_data_[i], (28, 28)), cmap='gray')
        a[0][i].set_xticks(())
        a[0][i].set_yticks(())
    plt.ion()


    saver = tf.train.Saver()
    # type = '_n_'+str(n_encoded)+'-batch'+str(BATCH_SIZE)+'-lr'+str(LearningRate)+'-'+str(XX)+'-'+str(SX1)+'-'+str(SX2)+'-'+str(RX)+'-'
    saver.restore(sess, '/Users/fengqi/Pycharm_py36/QF/4900000/_n_32-batch128-lr0.001-0.2-0.5-0.7-0.2-nx0' )

    LearningRate = 0.001
    ph_lr_ = np.ones(shape=[]) * LearningRate
    rangemax = 5000000;
    rangedvd = 100

    nx=0
    nx=1+nx
    type = '_n_' + str(n_encoded) + '-batch' + str(BATCH_SIZE) + '-lr' + str(LearningRate) + '-' + str(XX) + '-' + str(
        SX1) + '-' + str(SX2) + '-' + str(RX) + '-nx' + str(nx)
    for step in range(rangemax*nx,rangemax*(nx+1)):
        b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
        b_x, para = affine_transform1(b_x)

        _, loss_ = sess.run([train, loss], {tf_x: b_x, ph_encoded:ph_encoded_,ph_switch:ph_switch_,ph_lr:ph_lr_,ph_dis_e:ph_dis_e_})
        if step % 100 == 0:
            loss_ = sess.run(loss,{tf_x: test_x_, ph_encoded: ph_encoded_, ph_switch: ph_switch_, ph_lr: ph_lr_,
                                 ph_dis_e: ph_dis_e_})
            print('step:', step,'| train loss: %.6f' % loss_)


        if step%rangedvd==0:
            view_decoded_data, w = sess.run([decoded, weights],
                                                   {tf_x: view_data_,
                                                    ph_encoded: view_ph_encoded_,
                                                    ph_switch: view_ph_switch_,
                                                    ph_dis_e: view_ph_dis_e_})
            for i in range(N_TEST_IMG):
                a[1][i].imshow(np.reshape(view_decoded_data[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
                a[2][i].clear()
                a[2][i].imshow(np.reshape(w[:, i], (28, 28)), cmap='gray')
                a[2][i].set_xticks(());
                a[2][i].set_yticks(())
            plt.draw();plt.title(str('%.5f' % loss_));plt.savefig('./'+type+str(step)+'.png')
            if step%rangedvd%100==0:
                saver.save(sess, '/Users/fengqi/Pycharm_py36/QF/' + type)








a=1