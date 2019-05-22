
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from scipy import optimize

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
# encoder
en0b = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en0=tf.layers.batch_normalization(en0b)
en1b = tf.layers.dense(en0, 64, tf.nn.tanh)
en1=tf.layers.batch_normalization(en1b)
en2b = tf.layers.dense(en1, 32, tf.nn.tanh)
en2=tf.layers.batch_normalization(en2b)
en3b = tf.layers.dense(en2, 16, tf.nn.tanh)
en3=tf.layers.batch_normalization(en3b)
ff_encoded = tf.layers.dense(en3, n_encoded,tf.nn.tanh)
encodedb=ff_encoded*ph_switch+ph_encoded
encoded = tf.layers.batch_normalization(encodedb)
# decoder
de0b = tf.layers.dense(encoded, 16, tf.nn.tanh)
de0 = tf.layers.batch_normalization(de0b)
de1b = tf.layers.dense(de0, 32, tf.nn.tanh)
de1 = tf.layers.batch_normalization(de1b)
de2b = tf.layers.dense(de1, 64, tf.nn.tanh)
de2 = tf.layers.batch_normalization(de2b)
de3b = tf.layers.dense(de2, 128, tf.nn.tanh)
de3 = tf.layers.batch_normalization(de3b)
decoded = tf.layers.dense(de3, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = mnist.test.images[:N_TEST_IMG]
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(()); a[0][i].set_yticks(())

ph_encoded_ = np.zeros(shape=[BATCH_SIZE, n_encoded])
ph_encoded_view = np.zeros(shape=[N_TEST_IMG, n_encoded])
ph_switch_= np.ones(shape=[1])

saver = tf.train.Saver()
saver.restore(sess,'/Users/fengqi/Pycharm_py36/QF/Autoencoder')

a=1

num=4
lam=0
ff, aa = plt.subplots(num, num)
plt.ion()
for nn in range(num):
    for pp in range(num):

        #print(nn,pp)
        for step in range(100):
            b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
            encoded_, en3_ = sess.run([ encoded, encoded], {tf_x: b_x,ph_encoded:ph_encoded_,ph_switch:ph_switch_})
            if step==0:
                e_col=encoded_
                n_col=en3_[:,nn]
            else:
                e_col=np.append(e_col,encoded_,axis=0)
                n_col=np.append(n_col,en3_[:,nn],axis=0)

        sz1,sz2=e_col.shape
        e_col=np.append(np.ones([sz1,1]),e_col,axis=1)
        ee=np.zeros([sz1,(sz2+1)*(sz2+1)])

        for ii in range (n_encoded):
            for jj in range(n_encoded):
                ee[:,jj+ii*sz2]=np.multiply(e_col[:,ii],e_col[:,jj])
        C, _, _, _ = scipy.linalg.lstsq(ee, n_col)
        coeff_=np.reshape(C,[sz2+1,sz2+1])
        fun = lambda x: -np.dot(np.dot(x.T,coeff_),x)+lam*np.dot(x.T,x)
        bnds = ([1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1])
        res=optimize.minimize(fun,np.zeros([sz2+1,]),bounds=bnds)

        print(-np.dot(np.dot(res.x.T, coeff_), res.x),lam*np.dot(res.x.T,res.x))
        ph_switch_= np.ones(shape=[1])*0
        ph_encoded_ = res.x[1:].reshape([1,8])
        decoded_ = sess.run(decoded, {tf_x: np.zeros([1,784]), ph_encoded: ph_encoded_, ph_switch: ph_switch_})

        print(res.x[:5])

        aa[nn][pp].clear()
        aa[nn][pp].imshow(np.reshape(decoded_, (28, 28)), cmap='gray')
        plt.draw();
        plt.pause(0.01)
a=1


num=4
lam=0
ff, aa = plt.subplots(num, num)
plt.ion()
for nn in range(num):
    for pp in range(num):

        #print(nn,pp)
        for step in range(10):
            b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
            encoded_, en3_ = sess.run([ encoded, de0], {tf_x: b_x,ph_encoded:ph_encoded_,ph_switch:ph_switch_})
            if step==0:
                e_col=encoded_
                n_col=en3_[:,nn]
            else:
                e_col=np.append(e_col,encoded_,axis=0)
                n_col=np.append(n_col,en3_[:,nn],axis=0)

        sz1,sz2=e_col.shape
        e_col=np.append(np.ones([sz1,1]),e_col,axis=1)
        ee=np.zeros([sz1,(sz2+1)*(sz2+1)])

        for ii in range (n_encoded):
            for jj in range(n_encoded):
                ee[:,jj+ii*sz2]=np.multiply(e_col[:,ii],e_col[:,jj])
        C, _, _, _ = scipy.linalg.lstsq(ee, n_col)
        coeff_=np.reshape(C,[sz2+1,sz2+1])
        fun = lambda x: -np.dot(np.dot(x.T,coeff_),x)+lam*np.dot(x.T,x)
        bnds = ([1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1],[-1, 1])
        res=optimize.minimize(fun,np.zeros([sz2+1,]),bounds=bnds)

        print(-np.dot(np.dot(res.x.T, coeff_), res.x),lam*np.dot(res.x.T,res.x))
        ph_switch_= np.ones(shape=[1])*0
        ph_encoded_ = res.x[1:].reshape([1,8])
        decoded_ = sess.run(decoded, {tf_x: np.zeros([1,784]), ph_encoded: ph_encoded_, ph_switch: ph_switch_})

        print(res.x[:5])

        aa[nn][pp].clear()
        aa[nn][pp].imshow(np.reshape(decoded_, (28, 28)), cmap='gray')
        plt.draw();
        plt.pause(0.01)
a=1