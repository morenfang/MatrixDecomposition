import numpy as np
from numpy.linalg import *
from scipy import signal


# 2-dim matrix, k's size is singular, and k1/k2's size is k_size + 1 // 2.
def compute_k1(k, k2):
    k_size = k.shape[0]
    k1_size = (k.shape[0] + 1) // 2
    k2_size = k1_size
    k1 = np.zeros((k1_size, k1_size))
    # k2_shape = (k2_size, k2_size)
    # k_shape = (batch_size, out_channel, k_size, k_size)
    k2 = np.random.randn(k2_size, k2_size)
    # padding_k1 = k2_size - 1
    padding_k2 = k1_size - 1

    k2_pad = np.pad(k2, padding_k2, 'constant')
    m_p = np.zeros((k_size * k_size, k1_size * k1_size))

    for h in range(k_size):
        for w in range(k_size):
            for l in range(k1_size):
                for m in range(k1_size):
                    m_p[k_size * h + w][k1_size * l + m] = k2_pad[h + l][w + m]
    k1 = np.dot(np.dot(inv(np.dot(m_p.T, m_p)), m_p.T), k.reshape((k_size * k_size, 1))).reshape((k1_size, k1_size))

    return k1


def compute_k2(k, k1):
    k_size = k.shape[0]
    k1_size = (k.shape[0] + 1) // 2
    k2_size = k1_size
    k2 = np.ones((4, 4), dtype=np.float32)
    # k2_shape = (k2_size, k2_size)
    # k_shape = (batch_size, out_channel, k_size, k_size)
    k2 = np.random.randn(k2_size, k2_size)
    # padding_k1 = k2_size - 1
    padding_k2 = k1_size - 1

    k2_pad = np.pad(k2, padding_k2, 'constant')
    m_p = np.zeros((k_size * k_size, k1_size * k1_size))

    for h in range(k_size):
        for w in range(k_size):
            for l in range(k1_size):
                for m in range(k1_size):
                    m_p[k_size * h + w][k1_size * l + m] = k1[l][m] * k2_pad[h + l][w + m]
    k2 = np.dot(np.dot(inv(np.dot(m_p.T, m_p)), m_p.T), k.reshape((k_size * k_size, 1))).reshape((k1_size, k1_size))
    return k2


def test():
    # ka = np.random.rand(4, 4)
    # kb = np.random.rand(4, 4)
    k = np.random.randn(7, 7)
    # signal.convolve2d(ka, kb, mode='full', boundary='fill', fillvalue=0)# np.random.randn(7, 7)
    k2 = np.random.randn(4, 4)
    print(k2.shape, k2)
    k1 = compute_k1(k, k2)
    # k_swap = np.zeros((2, 4, 4))
    # k_swap[0] = k1
    # k_swap[1] = k2
    print(k2)
    for i in range(1000):
        # k_tmp = k_swap[1]
        # k_swap[1] = compute_k1(k, k_swap[0])
        # k_swap[0] = k_tmp
        k2 = compute_k2(k, k1)
        k1 = compute_k1(k, k2)
        k_val = signal.convolve2d(k1, k2, mode='full', boundary='fill', fillvalue=0)
        cof = np.corrcoef(k.reshape(1, -1), k_val.reshape(1, -1))
        s2 = np.sqrt(np.sum(np.square(k.reshape((1, -1)) - k_val.reshape(1, -1))))
        print('corrcoef: ', cof[0][1])
        print('L2: ', s2)
        # print(k_swap[1])
        # print(k_val)

    # print(k_val.shape, k_val)
    # print(k.shape, k)
    # print(k)


if __name__ == '__main__':
    # ka = np.random.rand(4, 4)
    # kb = np.random.rand(4, 4)
    # k = signal.convolve2d(ka, kb, mode='full', boundary='fill', fillvalue=0)  # np.random.randn(7, 7)
    # k1 = np.random.randn(4, 4)
    # print(compute_k2(k, k1))
    test()
