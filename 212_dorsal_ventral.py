
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os

def affine_transform2(b_x):
    xx = np.random.uniform(low=-.50, high=.50, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-.50, high=.50, size=1)
    sx = np.random.uniform(low=.2, high=1.5, size=1)
    sy = np.random.uniform(low=.21, high=1.5, size=1)
    rr = np.random.uniform(low=-0.5, high=0.5, size=1)
    sz1,sz2=b_x.shape
    images = b_x.reshape([sz1, 28, 28, 1])
    seq = iaa.Sequential([iaa.Affine(translate_percent={"x": xx, "y":yy},
                         scale={"x": sx, "y": sy},rotate=rr*180,)])
    images_aug = seq.augment_images(images)
    b_x=np.reshape(images_aug,[sz1,sz2])
    return b_x,np.concatenate((xx,yy,sx,sy,rr),axis=0)


tf.set_random_seed(1)

# Hyper Parameters
BATCH_SIZE = 64
N_TEST_IMG = 10
n_encoded=8
# Mnist digits
mnist = input_data.read_data_sets('./mnist', one_hot=False)     # use not one-hotted target data
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]



# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 28*28])    # value in the range of (0, 1)
ph_encoded = tf.placeholder(tf.float32, [None, n_encoded*2])
ph_switch=  tf.placeholder(tf.float32, [1])
ph_lr=tf.placeholder(tf.float32, [])
ph_dis_e=tf.placeholder(tf.float32, [None,n_encoded*2])
# encoder


en0d = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1d = tf.layers.dense(en0d, 64, tf.nn.tanh)
en2d = tf.layers.dense(en1d, 32, tf.nn.tanh)
en3d = tf.layers.dense(en2d, 16, tf.nn.tanh)

en0v = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1v = tf.layers.dense(en0v, 64, tf.nn.tanh)
en2v = tf.layers.dense(en1v, 32, tf.nn.tanh)
en3v = tf.layers.dense(en2v, 16, tf.nn.tanh)
en3=tf.concat([en3d,en3v],axis=1)
ff_encoded = tf.layers.dense(en3, n_encoded*2,tf.nn.tanh)

enc=ff_encoded*ph_switch+ph_encoded*(1-ph_switch)
encoded=tf.multiply(enc,ph_dis_e)

# decoder
de0 = tf.layers.dense(encoded, 16, tf.nn.tanh)
de1 = tf.layers.dense(de0, 32, tf.nn.tanh)
de2 = tf.layers.dense(de1, 64, tf.nn.tanh)
de3 = tf.layers.dense(de2, 128, tf.nn.tanh)
decoded = tf.layers.dense(de3, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(ph_lr).minimize(loss)
weightv = tf.get_default_graph().get_tensor_by_name(os.path.split(en0v.name)[0] + '/kernel:0')
weightd = tf.get_default_graph().get_tensor_by_name(os.path.split(en0d.name)[0] + '/kernel:0')

sess = tf.Session()
sess.run(tf.global_variables_initializer())



ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded*2])
view_ph_encoded = np.zeros(shape=[N_TEST_IMG, n_encoded*2])
ph_switch_= np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[BATCH_SIZE, n_encoded*2])


# saver = tf.train.Saver()
# saver.restore(sess,'/Users/fengqi/Pycharm_py36/QF/temp')

view_data = mnist.test.images[:N_TEST_IMG]
view_data_=view_data*0;
for i in range(N_TEST_IMG):
    view_data_[i,:], para = affine_transform2(np.reshape(view_data[i],[1,784]))

view_ph_switch_ = np.ones(shape=[1])
view_ph_dis_e_=np.ones(shape=[N_TEST_IMG, n_encoded*2])

f, a = plt.subplots(6, N_TEST_IMG)
for i in range(N_TEST_IMG):
    a[0][i].clear()
    a[0][i].imshow(np.reshape(view_data_[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())
plt.ion()

ph_lr_= np.ones(shape=[])*0.001
for step in range(6000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b_x, para = affine_transform2(b_x)

    _, loss_ ,wd_t,wv_t= sess.run([train, loss,weightd,weightv], {tf_x: b_x, ph_encoded:ph_encoded_,ph_switch:ph_switch_,ph_lr:ph_lr_,ph_dis_e:ph_dis_e_})
    if step % 100 == 0:     # plotting
        print('step:', step,'| train loss: %.4f' % loss_)
        view_decoded_data,wv1,wd1= sess.run([decoded,weightv,weightd],{tf_x: view_data_, ph_encoded: view_ph_encoded, ph_switch: view_ph_switch_,
                                           ph_dis_e: view_ph_dis_e_})
        if step%1000==0:
            for i in range(N_TEST_IMG):
                a[1][i].imshow(np.reshape(view_decoded_data[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
                a[2][i].clear()
                a[2][i].imshow(np.reshape(wv1[:, i], (28, 28)), cmap='gray')
                a[2][i].set_xticks(());
                a[2][i].set_yticks(())
                a[3][i].clear()
                a[3][i].imshow(np.reshape(wd1[:,i], (28, 28)), cmap='gray')
                a[3][i].set_xticks(());
                a[3][i].set_yticks(())
            plt.draw();plt.pause(0.02)

            if step == 0:
                wd = np.reshape(wd_t,[784,128,1])
                wv = np.reshape(wv_t, [784, 128, 1])
            else:
                wd = np.append(wd, np.reshape(wd_t,[784,128,1]), axis=2)
                wv = np.append(wv, np.reshape(wv_t,[784,128,1]), axis=2)


N_TEST_IMG=10;kkk=np.random.randint(1, 128)
f, a = plt.subplots(6, N_TEST_IMG)
view_ph_encoded = np.zeros(shape=[N_TEST_IMG, n_encoded*2])
ph_dis_e_ = np.ones(shape=[N_TEST_IMG, n_encoded*2])

decoded_data,wv_,wd_ = sess.run([decoded,weightv,weightd],
                        {tf_x: view_data_, ph_encoded: view_ph_encoded, ph_switch: ph_switch_, ph_dis_e: ph_dis_e_})
vvv=np.random.randint(1, 128)
ddd=np.random.randint(1, 128)

for i in range(N_TEST_IMG):
    a[0][i].clear()
    a[0][i].imshow(np.reshape(view_data_[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())
    a[1][i].clear()
    a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())

    a[2][i].clear()
    a[2][i].imshow(np.reshape(wv_[:,np.random.randint(1, 128)], (28, 28)), cmap='gray')
    a[2][i].set_xticks(());
    a[2][i].set_yticks(())
    a[3][i].clear()
    a[3][i].imshow(np.reshape(wd_[:,np.random.randint(1, 128)], (28, 28)), cmap='gray')
    a[3][i].set_xticks(());
    a[3][i].set_yticks(())

    a[4][i].clear()
    a[4][i].imshow(np.reshape(wv[:,vvv,i], (28, 28)), cmap='gray')
    a[4][i].set_xticks(());
    a[4][i].set_yticks(())
    a[5][i].clear()
    a[5][i].imshow(np.reshape(wd[:,ddd,i], (28, 28)), cmap='gray')
    a[5][i].set_xticks(());
    a[5][i].set_yticks(())
plt.ion()
plt.draw();
plt.pause(0.01)







saver = tf.train.Saver()
saver.save(sess,'/Users/fengqi/Pycharm_py36/QF/dural')


