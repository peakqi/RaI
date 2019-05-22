
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

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
en3b = tf.layers.dense(en1, 16, tf.nn.tanh)
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
de3b = tf.layers.dense(de1, 128, tf.nn.tanh)
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
#saver.restore(sess,'/Users/fengqi/Pycharm_py36/QF/Autoencoder')
for step in range(15000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], {tf_x: b_x,ph_encoded:ph_encoded_,ph_switch:ph_switch_})

    if step % 100 == 0:     # plotting

        print('step:', step,'| train loss: %.4f' % loss_)
        # plotting decoded image (second row)
        decoded_data = sess.run(decoded, {tf_x:view_data,ph_encoded:ph_encoded_view,ph_switch:ph_switch_})

        if 0:
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.01)
saver = tf.train.Saver()
saver.save(sess,'/Users/fengqi/Pycharm_py36/QF/Autoencoder')

plt.ioff()

