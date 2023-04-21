import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ipywidgets import interact
import matplotlib.animation as animation

class parameters(object):
    numbeers_of_x = 200
    numbeers_of_y = 200
    temerature = 2.5
    time_steps = 500
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
    total_mean_energy = np.mean(total_energy_list)
    total_squared_mean_energy = np.mean(np.array(total_energy_list)**2)
    return field, total_mean_energy, total_squared_mean_energy

def _ising_update(field, n, m, beta):
    total = 0 #初始化相鄰總能量
    N, M = field.shape #邊界
    for i in range(n-1, n+2): #加總相鄰左右的能量
        for j in range(m-1, m+2): #加總相鄰上下的能量
            if i == n and j == m: #跳過本身
                continue
            total += field[i % N, j % M] #計算相鄰總能量狀態，循環邊界條件
    dH = - par.j * field[n, m] * total #計算 hamitonian 變化
    if dH >= 0: #系統往能量較低處演化
        field[n, m] *= -1
    elif np.exp(dH * beta) > np.random.rand(): #根據 boltzmann factor 機率性反轉
        field[n, m] *= -1
    return dH

base = random_spin_field(par.numbeers_of_x, par.numbeers_of_y)
img, _, __ = ising_step(base)
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
    img, mean_energy, _ = ising_step(images[-1])
    images.append(img.copy())
    mean_energy_list.append(mean_energy)

# display_ising_sequence(images)

# for i in range(len(images)):
#     plt.imshow(images[i])
#     plt.pause(0.01)
# plt.show()

plot_animation(images, save=False)
plot_mean_energy_with_time_steps(mean_energy_list, save=False)

# find the phase transition
mean_energy_with_diff_number = []
mean_magnetic_moment_with_diff_number = []
mean_capacity_with_diff_number = []
flag = 0
size_number_start = 5
size_number_end = 26
space = 10
temperature_start = 0.0001
temperature_end = 5*par.j
temperature_spot_numbers = 20
for number in range(size_number_start, size_number_end, space):
    globals()["mean_energy_with_number" + str(number)] = []
    globals()["mean_magnetic_moment_with_number" + str(number)] = []
    globals()["mean_capacity_with_number" + str(number)] = []
    mean_energy_with_diff_number.append(globals()["mean_energy_with_number" + str(number)])
    mean_magnetic_moment_with_diff_number.append(globals()["mean_magnetic_moment_with_number" + str(number)])
    mean_capacity_with_diff_number.append(globals()["mean_capacity_with_number" + str(number)])
    for temperature in np.linspace(temperature_start, temperature_end, temperature_spot_numbers):
        beta = 1 / temperature
        images = [random_spin_field(number, number)]
        mean_energy_list = []
        squared_mean_energy_list = []
        for i in range(par.time_steps):
            img, mean_energy, squared_mean_energy = ising_step(images[-1], beta=beta)
            images.append(img.copy())
            mean_energy_list.append(mean_energy)
            squared_mean_energy_list.append(squared_mean_energy)
        mean_energy_with_diff_number[flag].append(mean_energy_list[-1])
        mean_capacity_with_diff_number[flag].append((beta**2)*(squared_mean_energy_list[-1]-mean_energy_list[-1]**2))
        w, h = images[-1].shape
        temp_magnetic_moment_list = []
        for i in range(w):
            for j in range(h):
                temp_magnetic_moment_list.append(images[-1][i][j])
        mean_magnetic_moment_with_diff_number[flag].append(np.abs(np.mean(temp_magnetic_moment_list)))

    flag += 1

plot_used_list = [i for i in range(size_number_start, size_number_end, space)]
plot_used_list_2 = ['r*', 'gp', 'bd']
plot_used_list_3 = ['r--', 'g--', 'b--']
for i in range(len(plot_used_list)):
    plt.plot(np.linspace(temperature_start, temperature_end, temperature_spot_numbers),
             mean_energy_with_diff_number[i],plot_used_list_2[i],
             label="size = " + str(plot_used_list[i]**2))
    plt.plot(np.linspace(temperature_start, temperature_end, temperature_spot_numbers),
             mean_energy_with_diff_number[i], plot_used_list_3[i])
plt.title(r"$\langle E \rangle$-T with different size ")
plt.xlabel("T")
plt.ylabel(r"$\langle E \rangle$")
plt.legend()
plt.savefig("./mean_energy-temperature_with_diff_number.png")
plt.show()

for i in range(len(plot_used_list)):
    plt.plot(np.linspace(temperature_start, temperature_end, temperature_spot_numbers),
             mean_capacity_with_diff_number[i],plot_used_list_2[i],
             label="size = " + str(plot_used_list[i]**2))
    plt.plot(np.linspace(temperature_start, temperature_end, temperature_spot_numbers),
             mean_capacity_with_diff_number[i], plot_used_list_3[i])
plt.title(r"$\langle C \rangle$-T with different size ")
plt.xlabel("T")
plt.ylabel(r"$\langle C \rangle$")
plt.legend()
plt.savefig("./mean_capacity-temperature_with_diff_number.png")
plt.show()

for i in range(len(plot_used_list)):
    plt.plot(np.linspace(temperature_start, temperature_end, temperature_spot_numbers),
             mean_magnetic_moment_with_diff_number[i],plot_used_list_2[i],
             label="size = " + str(plot_used_list[i]**2))
    plt.plot(np.linspace(temperature_start, temperature_end, temperature_spot_numbers),
             mean_magnetic_moment_with_diff_number[i], plot_used_list_3[i])
plt.title(r"$\langle m \rangle$-T with different size ")
plt.xlabel("T")
plt.ylabel(r"$\langle m \rangle$")
plt.legend()
plt.savefig("./mean_magnetic_moment-temperature_with_diff_number.png")
plt.show()