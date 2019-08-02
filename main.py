import numpy as np
from numpy.linalg import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as image
import tensorflow as tf

np.random.seed(2)
m_k = np.random.randn(1024, 7, 7)
m_k_val = np.zeros((7, 7))
m_k1 = np.zeros((1024, 4, 4))
np.random.seed(2)
m_k2 = np.random.randn(4, 4)
m_p = np.zeros((49, 16))

m_k_1_49 = m_k.reshape((1024, 49))
m_k1_1_16 = m_k1.reshape((1024, 16))
m_k2_pad = np.pad(m_k2, (3, 3), 'constant')

img_path = "./sample.jpeg"


def main():
    with open('./sample.txt') as file_large:
        matrix_large = file_large.read()
        # print(matrix_large)
    # print(m_k2)

    # prepare m_p
    # cof = np.zeros((49, 16))
    np.set_printoptions(precision=5, suppress=True)
    for i in range(7):
        for j in range(7):
            for l in range(4):
                for m in range(4):
                    m_p[7 * i + j][4 * l + m] = m_k2_pad[i + l][j + m]
    print(m_p)
    for i in range(1024):
        m_k1[i, :, :] = np.dot(np.dot(inv(np.dot(m_p.T, m_p)), m_p.T), m_k[i, :, :].reshape((49, 1))).reshape((4, 4))
    # print(np.dot(m_p, m_k2.T))
    # print(m_k1)
    # print(m_k2)
    '''
    for ch in range(1024):
        for i in range(7):
            for j in range(7):
                for l in range(4):
                    for m in range(4):
                        m_k_val[i][j] += m_k2_pad[i + l][j + m] * m_k1[ch][l][m]
        l2 = np.sqrt(np.sum(np.square(m_k[ch].reshape((1, -1)) - m_k_val.reshape(1, -1))))
        print(l2)
    '''

    img = image.imread(img_path)
    np.random.seed(3)
    img_1 = np.random.randn(1024, 14, 14)  # img[:,:,0]
    img_1_pad = np.pad(img_1, ((0, 0), (3, 3), (3, 3)), 'constant')

    # plt.imshow(img_1)
    # plt.show()

    # print(img_1.shape)

    out_img_1 = np.zeros((1024, 14, 14))
    out_img_2 = np.zeros((1024, 14, 14))
    for ch in range(1024):
        for i in range(14):
            for j in range(14):
                for l in range(7):
                    for m in range(7):
                        out_img_1[ch][i][j] += img_1_pad[ch][i + l][j + m] * m_k[ch][l][m]
    # plt.imshow(out_img_1)
    # plt.show()
    img_1_pad = np.pad(img_1, ((0, 0), (1, 2), (1, 2)), 'constant')

    out_img_tmp = np.zeros((1024, 14, 14))
    for ch in range(1024):
        for i in range(14):
            for j in range(14):
                for l in range(4):
                    for m in range(4):
                        out_img_tmp[ch][i][j] += img_1_pad[ch][i + l][j + m] * m_k[ch][l][m]
    img_1_pad = np.pad(out_img_tmp, ((0, 0), (1, 2), (1, 2)), 'constant')
    for ch in range(1024):
        for i in range(14):
            for j in range(14):
                for l in range(4):
                    for m in range(4):
                        out_img_2[ch][i][j] += img_1_pad[ch][i + l][j + m] * m_k[ch][l][m]

    print(out_img_1.reshape((1, -1)).shape)
    print(out_img_2.reshape((1, -1)).shape)

    s2 = np.sqrt(np.sum(np.square(out_img_2.reshape((1, -1)) - out_img_1.reshape(1, -1))))
    co = np.corrcoef(out_img_2.reshape((1, -1)), out_img_1.reshape(1, -1))
    print(s2)
    print(co)

    '''
    in_img = tf.placeholder(tf.float32, shape=[1, 14, 14, 1024])
    in_img2 = tf.placeholder(tf.float32, shape=[1, 14, 14, 1024])
    kernel_1 = tf.placeholder(tf.float32, shape=[1024, 4, 4, 1])
    kernel_2 = tf.placeholder(tf.float32, shape=[1024, 4, 4, 1])

    with tf.Session() as sess:
        conv2d_op = tf.nn.conv2d(in_img, kernel_1, strides=[1, 1, 1, 1], padding="SAME", data_format='NHWC')
        result_1 = sess.run(conv2d_op, feed_dict={in_img: img_1.reshape([1, 14, 14, 1024]),
                                                  kernel_1: m_k1.reshape([4, 4, 1024, 1])})
        conv2d_op = tf.nn.conv2d(in_img2, kernel_2, strides=[1, 1, 1, 1], padding="SAME", data_format='NHWC')
        result_2 = sess.run(conv2d_op, feed_dict={in_img2: result_1,
                                                  kernel_2: (np.tile(m_k2, (1024, 1)).reshape([4, 4, 1024, 1]))})
        # print(result_2.shape)
        out_img_2 = result_2.reshape((1024, 14, 14))
        # print(out_img_1)
        # print(out_img_2)
        # print(out_img_2 - out_img_1)
        # print(sum(out_img_2 - out_img_1))
        # plt.imshow(result_2.reshape((250, 250)))
        # plt.show()

        s2 = np.sqrt(np.sum(np.square(out_img_2.reshape((1, -1)) - out_img_1.reshape(1, -1))))

        print(s2)
    '''
    # conv_test1 = nn.Conv2d(1, 1, (7, 7), 2, bias=False)
    # conv_test2 = nn.Conv2d(1, 1, (4, 4), 1, bias=False)
    # conv_test3 = nn.Conv2d(1, 1, (4, 4), 1, bias=False)


if __name__ == '__main__':
    main()
