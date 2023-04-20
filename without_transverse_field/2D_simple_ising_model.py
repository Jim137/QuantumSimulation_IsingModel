import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact
import matplotlib.animation as animation

class parameters(object):
    numbeers_of_x = 200
    numbeers_of_y = 200
    temerature = 2.5
    time_steps = 50
    j = 2

par = parameters()
beta=1/par.temerature

def random_spin_field(N, M):
    return np.random.choice([-1, 1], size=(N, M))

print(random_spin_field(10,10))

def display_spin_field(field):
    return Image.fromarray(np.uint8((field + 1) * 0.5 * 255))  # 0 ... 255

plt.imshow(display_spin_field(random_spin_field(par.numbeers_of_x, par.numbeers_of_y)))
plt.show()

def ising_step(field, beta = beta):
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    _ising_update(field, n, m, beta)
    return field

def _ising_update(field, n, m, beta):
    total = 0 #初始化相鄰總能量
    N, M = field.shape #邊界
    for i in range(n-1, n+2): #加總相鄰左右的能量
        for j in range(m-1, m+2): #加總相鄰上下的能量
            if i == n and j == m: #跳過本身
                continue
            total += field[i % N, j % M] #計算相鄰總能量，循環邊界條件
    dH = - par.j * field[n, m] * total #計算 hamitonian 變化
    if dH >= 0: #系統網能量較低處演化
        field[n, m] *= -1
    elif np.exp(dH * beta) > np.random.rand(): #根據 boltzmann factor 機率性反轉
        field[n, m] *= -1

plt.imshow(display_spin_field(ising_step(random_spin_field(par.numbeers_of_x, par.numbeers_of_y))))
plt.show()

# def display_ising_sequence(images):
#     def _show(frame=(0, len(images) - 1)):
#         return display_spin_field(images[frame])
#     return interact(_show)

images = [random_spin_field(par.numbeers_of_x, par.numbeers_of_y)]
for i in range(par.time_steps):
    images.append(ising_step(images[-1].copy()))

# display_ising_sequence(images)

# for i in range(len(images)):
#     plt.imshow(images[i])
#     plt.pause(0.01)
# plt.show()

fig, ax = plt.subplots()
ims = [[ax.imshow(im)] for im in images]
ani = animation.ArtistAnimation(fig, ims, interval=len(images), blit=True,
                                repeat_delay=1000)
# To save the animation, use e.g.
#
ani.save("./2D_ising_model.gif")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.gif", writer=writer)
plt.show()