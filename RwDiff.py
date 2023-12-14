from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def uniRW1D(N):
    u = 2 * randint(0, 2, size=N) - 1
    x = cumsum(u)
    return x


def gammaRW1D(N, shp, scale):
    u = np.random.gamma(shp, scale, N)
    x = cumsum(u)
    return x


def gaussRW1D(N, mn, dev):
    u = np.random.normal(mn, dev, N)
    x = cumsum(u)
    return x


def uniRW2D(N):
    u = 2 * randint(0, 2, size=(N, 2)) - 1
    x = cumsum(u[:, 0])
    y = cumsum(u[:, 1])
    return x, y


def uniDev1D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    dev = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(2 * randint(0, 2, size=N) - 1)
            xdev = std(x) ** 2
            dev[i] = dev[i] + xdev
        dev[i] = dev[i] / (M * 1.0)
    return dev, Nvalues


def uniDev2D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    devX = zeros(len(Nvalues))
    devY = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(2 * randint(0, 2, size=N) - 1)
            y = cumsum(2 * randint(0, 2, size=N) - 1)
            xdev = std(x) ** 2
            ydev = std(y) ** 2
            devX[i] += xdev
            devY[i] += ydev
        devX[i] = devX[i] / (M * 1.0)
        devY[i] = devY[i] / (M * 1.0)
    r = devX+devY
    return r, Nvalues


def uniDev3D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    devX = zeros(len(Nvalues))
    devY = zeros(len(Nvalues))
    devZ = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(2 * randint(0, 2, size=N) - 1)
            y = cumsum(2 * randint(0, 2, size=N) - 1)
            z = cumsum(2 * randint(0, 2, size=N) - 1)
            xdev = std(x) ** 2
            ydev = std(y) ** 2
            zdev = std(z) ** 2
            devX[i] += xdev
            devY[i] += ydev
            devZ[i] += zdev
        devX[i] = devX[i] / (M * 1.0)
        devY[i] = devY[i] / (M * 1.0)
        devZ[i] = devZ[i] / (M * 1.0)
    r = devX+devY+devZ
    return r, Nvalues


def gaussDev1D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    dev = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(np.random.normal(0, 1, N))
            xdev = std(x) ** 2
            dev[i] = dev[i] + xdev
        dev[i] = dev[i] / (M * 1.0)
    return dev, Nvalues


def gaussDev2D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    devX = zeros(len(Nvalues))
    devY = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(np.random.normal(0, 1, N))
            y = cumsum(np.random.normal(0, 1, N))
            xdev = std(x) ** 2
            ydev = std(y) ** 2
            devX[i] += xdev
            devY[i] += ydev
        devX[i] = devX[i] / (M * 1.0)
        devY[i] = devY[i] / (M * 1.0)
    r = devX+devY
    return r, Nvalues


def gaussDev3D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    devX = zeros(len(Nvalues))
    devY = zeros(len(Nvalues))
    devZ = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(np.random.normal(0, 1, N))
            y = cumsum(np.random.normal(0, 1, N))
            z = cumsum(np.random.normal(0, 1, N))
            xdev = std(x) ** 2
            ydev = std(y) ** 2
            zdev = std(z) ** 2
            devX[i] += xdev
            devY[i] += ydev
            devZ[i] += zdev
        devX[i] = devX[i] / (M * 1.0)
        devY[i] = devY[i] / (M * 1.0)
        devZ[i] = devZ[i] / (M * 1.0)
    r = devX+devY+devZ
    return r, Nvalues


def gammaDev1D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    dev = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(np.random.gamma(.1, .01, N))
            xdev = std(x) ** 2
            dev[i] = dev[i] + xdev
        dev[i] = dev[i] / (M * 1.0)
    return dev, Nvalues


def gammaDev2D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    devX = zeros(len(Nvalues))
    devY = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(np.random.gamma(.1, .01, N))
            y = cumsum(np.random.gamma(.1, .01, N))
            xdev = std(x) ** 2
            ydev = std(y) ** 2
            devX[i] = devX[i] + xdev
            devY[i] = devY[i] + ydev
        devX[i] = devX[i] / (M * 1.0)
        devY[i] = devY[i] / (M * 1.0)
    r = devX + devY
    return r, Nvalues


def gammaDev3D(N, M):
    Nvalues = [2 ** i for i in range(N)]
    devX = zeros(len(Nvalues))
    devY = zeros(len(Nvalues))
    devZ = zeros(len(Nvalues))
    for i in range(len(Nvalues)):
        N = Nvalues[i]
        for _ in range(M):
            x = cumsum(np.random.gamma(.1, .01, N))
            y = cumsum(np.random.gamma(.1, .01, N))
            z = cumsum(np.random.gamma(.1, .01, N))
            xdev = std(x) ** 2
            ydev = std(y) ** 2
            zdev = std(z) ** 2
            devX[i] = devX[i] + xdev
            devY[i] = devY[i] + ydev
            devZ[i] = devZ[i] + zdev
        devX[i] = devX[i] / (M * 1.0)
        devY[i] = devY[i] / (M * 1.0)
        devZ[i] = devZ[i] / (M * 1.0)
    r = devX + devY + devZ
    return r, Nvalues


def uniPosDis(N, M, D):
    x = zeros((M, D))  # change this
    for i in range(M):
        z = randint(3, size=(N, D)) - 1
        x[i, :] = sum(z, axis=0)
    return x


def gaussPosDis(N, M, D):
    x = zeros((M, D))
    for i in range(M):
        z = np.random.normal(0, 1, size=(N, D))
        x[i, :] = sum(z, axis=0)
    return x


def gammaPosDis(N, M, D):
    x = zeros((M, D))
    for i in range(M):
        z = np.random.gamma(.1, .01, size=(N, D))
        x[i, :] = sum(z, axis=0)
    return x


def lagDifUni(N, M, D, p):
    x = zeros((N, M, D))
    for i in range(N):
        for j in range(M):
            for k in range(D):
                if i == 0:
                    x[i, j, k] += -1*(rand(1) < p)+1*(rand(1) > (1-p))
                else:
                    x[i, j, k] = x[i-1, j, k] + -1 * (rand(1) < p) + 1 * (rand(1) > (1 - p))
    return x


def eulDif(N, M, D, p):  # for now use M and D as X and Y
    x = zeros((N, M, D))
    x[0, :, :] = 100
    for i in range(1, N):
        for j in range(1, M-1):
            for k in range(1, D-1):
                x[i, j, k] = x[i-1, j, k] + p * (-4*x[i-1, j, k] + x[i-1, j+1, k] + x[i-1, j-1, k] + x[i-1, j, k+1] + x[i-1, j, k-1])
    return x


def heavySide(N, M):
    x = arange(1, M+1)
    for i in range(M):
        z = (2*randint(0, 2, size=N) - 1)/sqrt(N)
        x[i] = sum(z)
    x = x / sqrt(N)
    return x


def green(k, t, x):
    return (1/sqrt(4*pi*k*t))*exp((-(x**2))/(4*k*t))


M0 = 10000
N0 = 10000
k = .005
est = heavySide(N0, M0)
x1 = hist(est)
ex = zeros(len(x1[0]))
for i in range(len(x1[0])):
    ex[i] = green(k, N0, x1[1][i])
err = abs(ex-(x1[0]/sum(x1[0])))/ex
# print(x1[1])
# print(ex)
# print(x1)
# plt.hist(ex)
# plt.show()

# print(ex)
# print(est)
# a = linspace(-.001, .001, 100)
# u = green(k, 10000, a)
# plt.plot(a, u)
# plt.show()
print(ex-(x1[0]/sum(x1[0])))
print(err)
print(average(err))

'''
def update(frame):
    ax.clear()
    hist = ax.hist2d(X[:, 0, frame], X[:, 1, frame], bins=(75, 75), cmap='inferno')


N0 = 100
M0 = 100000
D0 = 2
X = zeros((M0, D0, N0))

for i0 in range(N0):
    X[:, :, i0] = uniPosDis(i0, M0, D0)

fig, ax = plt.subplots()

hist = ax.hist2d([], [], bins=(75, 75), cmap='inferno')

animation = FuncAnimation(fig, update, frames=N0, interval=100, repeat=True)

animation.save('UniformP(x,N).gif', writer='pillow')

plt.show()
'''


# X = gaussPosDis(2, 100000, 2)

# plt.hist2d(X[:, 0], X[:, 1], bins=(75, 75), cmap='inferno')

# plt.colorbar()

# plt.show()


# Gamma shape .1 scale .01
# Gaussian mean 0 dev 1
# M=10000
