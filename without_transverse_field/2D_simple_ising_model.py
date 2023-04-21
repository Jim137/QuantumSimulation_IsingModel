import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact
import matplotlib.animation as animation

class parameters(object):
    numbeers_of_x = 200
    numbeers_of_y = 200
    temerature = 2.5
    time_steps = 100
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
    total_energy_list = []
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    dE = _ising_update(field, n, m, beta)
                    total_energy_list.append(dE)
    total_energy = np.mean(total_energy_list)
    return field, total_energy

def _ising_update(field, n, m, beta):
    total = 0 #初始化相鄰總能量
    N, M = field.shape #邊界
    for i in range(n-1, n+2): #加總相鄰左右的能量
        for j in range(m-1, m+2): #加總相鄰上下的能量
            if i == n and j == m: #跳過本身
                continue
            total += field[i % N, j % M] #計算相鄰總能量狀態，循環邊界條件
    dH = - par.j * field[n, m] * total #計算 hamitonian 變化
    if dH >= 0: #系統網能量較低處演化
        field[n, m] *= -1
    elif np.exp(dH * beta) > np.random.rand(): #根據 boltzmann factor 機率性反轉
        field[n, m] *= -1
    return dH

base = random_spin_field(par.numbeers_of_x, par.numbeers_of_y)
img, _ = ising_step(base)
plt.imshow(display_spin_field(img))
plt.show()

def plot_animation(images, save = True):
    fig, ax = plt.subplots()
    ims = [[ax.imshow(im)] for im in images]
    ani = animation.ArtistAnimation(fig, ims, interval=len(images), blit=True,
                                    repeat_delay=1000)
    # To save the animation, use e.g.
    #
    if save:
        ani.save("./2D_ising_model.gif")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.gif", writer=writer)
    plt.show()

def plot_mean_energy_with_time_steps(mean_energy_list, save = True):
    plt.plot([i for i in range(len(mean_energy_list))], mean_energy_list)
    plt.title("mean energy-time of 2D ising model")
    plt.xlabel("time steps")
    plt.ylabel("energy")
    plt.grid()
    if save:
        plt.savefig("./mean_energy_with_time_steps_2D_ising_model")
    plt.show()

# def display_ising_sequence(images):
#     def _show(frame=(0, len(images) - 1)):
#         return display_spin_field(images[frame])
#     return interact(_show)

images = [random_spin_field(par.numbeers_of_x, par.numbeers_of_y)]
mean_energy_list = []
for i in range(par.time_steps):
    img, mean_energy = ising_step(images[-1])
    images.append(img.copy())
    mean_energy_list.append(mean_energy)

# display_ising_sequence(images)

# for i in range(len(images)):
#     plt.imshow(images[i])
#     plt.pause(0.01)
# plt.show()

plot_animation(images, save=False)
plot_mean_energy_with_time_steps(mean_energy_list, save=True)

# find the phase transition
mean_energy_with_diff_number = []
mean_magnetic_moment_with_diff_number = []
flag = 0
size_number_start = 4
size_number_end = 21
space = 6
for number in range(size_number_start, size_number_end, space):
    globals()["mean_energy_with_number" + str(number)] = []
    globals()["mean_magnetic_moment_with_number" + str(number)] = []
    mean_energy_with_diff_number.append(globals()["mean_energy_with_number" + str(number)])
    mean_magnetic_moment_with_diff_number.append(globals()["mean_magnetic_moment_with_number" + str(number)])
    for temperature in np.linspace(0.2, 4 * par.j, 4 * par.j * 5):
        beta = 1 / temperature
        images = [random_spin_field(number, number)]
        mean_energy_list = []
        for i in range(par.time_steps):
            img, mean_energy = ising_step(images[-1], beta=beta)
            images.append(img.copy())
            mean_energy_list.append(mean_energy)
        mean_energy_with_diff_number[flag].append(mean_energy_list[-1])
        w, h = images[-1].shape
        temp_magnetic_moment_list = []
        for i in range(w):
            for j in range(h):
                temp_magnetic_moment_list.append(images[-1][i][j])
        mean_magnetic_moment_with_diff_number[flag].append(np.abs(np.mean(temp_magnetic_moment_list)))

    flag += 1

plot_used_list = [i for i in range(size_number_start, size_number_end, space)]
for i in range(len(plot_used_list)):
    plt.plot(np.linspace(0, 4 * par.j, 4 * par.j * 5), mean_energy_with_diff_number[i],
             label="size = " + str(plot_used_list[i]**2))
plt.title("E-T with different size ")
plt.xlabel("T")
plt.ylabel("E")
plt.legend()
plt.savefig("./mean_energy-temperature_with_diff_number.png")
plt.show()

for i in range(len(plot_used_list)):
    plt.plot(np.linspace(0, 4 * par.j, 4 * par.j * 5), mean_magnetic_moment_with_diff_number[i],
             label="size = " + str(plot_used_list[i]**2))
plt.title("m-T with different size ")
plt.xlabel("T")
plt.ylabel("m")
plt.legend()
plt.savefig("./mean_magnetic_moment-temperature_with_diff_number.png")
plt.show()