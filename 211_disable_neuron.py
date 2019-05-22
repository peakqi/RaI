
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from imgaug import augmenters as iaa
import os

def affine_transform2(b_x):
    xx = np.random.uniform(low=-.0, high=0, size=1)  # (-0.5,0.5)
    yy = np.random.uniform(low=-0, high=0, size=1)
    sx = np.random.uniform(low=1, high=1, size=1)
    sy = np.random.uniform(low=1, high=1, size=1)
    rr = np.random.uniform(low=-0, high=0, size=1)
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
LR = 0.002         # learning rate
N_TEST_IMG = 10
n_encoded=8
# Mnist digits
mnist = input_data.read_data_sets('./mnist', one_hot=False)     # use not one-hotted target data
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]



# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 28*28])    # value in the range of (0, 1)
ph_encoded = tf.placeholder(tf.float32, [None, n_encoded])
ph_switch=  tf.placeholder(tf.float32, [1])
ph_lr=tf.placeholder(tf.float32, [])
ph_dis_e=tf.placeholder(tf.float32, [None,n_encoded])
# encoder


en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 32, tf.nn.tanh)
en3 = tf.layers.dense(en2, 16, tf.nn.tanh)
ff_encoded = tf.layers.dense(en3, n_encoded,tf.nn.tanh)
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
weights = tf.get_default_graph().get_tensor_by_name(os.path.split(en0.name)[0] + '/kernel:0')

sess = tf.Session()
sess.run(tf.global_variables_initializer())



ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded])
ph_encoded_view = np.zeros(shape=[N_TEST_IMG, n_encoded])
ph_switch_= np.ones(shape=[1])
ph_dis_e_ = np.ones(shape=[BATCH_SIZE, n_encoded])



ph_lr_= np.ones(shape=[])*0.001
saver = tf.train.Saver()
saver.restore(sess,'/Users/fengqi/Pycharm_py36/QF/temp')
for step in range(0):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    b_x, para = affine_transform2(b_x)

    _, loss_ ,w_= sess.run([train, loss,weights], {tf_x: b_x, ph_encoded:ph_encoded_,ph_switch:ph_switch_,ph_lr:ph_lr_,ph_dis_e:ph_dis_e_})
    if step % 100 == 0:     # plotting
        print('step:', step,'| train loss: %.4f' % loss_)
        if step % 300 == 0:
            if step == 0:
                w1 = np.reshape(w_,[784,128,1])
            else:
                w1 = np.append(w1, np.reshape(w_,[784,128,1]), axis=2)

#
# saver = tf.train.Saver()
# saver.save(sess,'/Users/fengqi/Pycharm_py36/QF/temp')
N_TEST_IMG=20

ph_encoded_view = np.zeros(shape=[N_TEST_IMG, n_encoded])
view_data = mnist.test.images[:N_TEST_IMG]
view_data_, para = affine_transform2(view_data)
ph_dis_e_ = np.zeros(shape=[N_TEST_IMG, n_encoded])


nn=5

f, a = plt.subplots(nn+1, N_TEST_IMG)
plt.ion()
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data_[i], (28, 28)), cmap='gray')
for j in range(nn):
    for i in range(N_TEST_IMG):
        ph_dis_e_[:, 0] = j*0.2
        decoded_data = sess.run(decoded,{tf_x: view_data_, ph_encoded: ph_encoded_view, ph_switch: ph_switch_, ph_dis_e: ph_dis_e_})
        a[j+1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
plt.draw();plt.savefig('/Users/fengqi/n0.png')



ph_encoded_view = np.ones(shape=[N_TEST_IMG, n_encoded])
view_data = mnist.test.images[:N_TEST_IMG]
view_data_, para = affine_transform2(view_data)
ph_dis_e_ = np.ones(shape=[N_TEST_IMG, n_encoded])
f, a = plt.subplots(nn+1, N_TEST_IMG)
plt.ion()
ori_img=np.zeros([N_TEST_IMG,784])
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data_[i], (28, 28)), cmap='gray');a[0][i].set_xticks(()); a[0][i].set_yticks(())
for j in range(nn):
    for i in range(N_TEST_IMG):
        ph_dis_e_[:, :] = j*0.2
        decoded_data = sess.run(decoded,{tf_x: view_data_, ph_encoded: ph_encoded_view, ph_switch: ph_switch_, ph_dis_e: ph_dis_e_})
        if j==0:
            a[j + 1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
            ori_img[i]=decoded_data[i]
        else:
            a[j+1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
            ori_img[i] = decoded_data[i];
        a[j+1][i].set_xticks(());a[j+1][i].set_yticks(())

plt.draw();plt.savefig('/Users/fengqi/n0.png')




f, a = plt.subplots(6, N_TEST_IMG)
plt.ion()
for i in range(8*16):
    plt.subplot(8, 16,i+1)
    plt.subplot(8, 16,i+1).set_xticks(()); plt.subplot(8, 16,i+1).set_yticks(())
    plt.imshow(np.reshape(w_[:,i], [28, 28]), cmap='gray')
plt.draw()

a=1





saver = tf.train.Saver()
saver.save(sess,'/Users/fengqi/Pycharm_py36/QF/Autoencoder')


