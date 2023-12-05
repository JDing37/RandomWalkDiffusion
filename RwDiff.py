from pylab import *
import matplotlib.pyplot as plt


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
    return devX, devY, Nvalues


def uniPosDisNorm(N, M, D):
    x = zeros((M, D))
    n = zeros(M)
    for i in range(M):
        z = 2 * randint(2, size=(N, D)) - 1
        x[i, :] = sum(z, axis=0)
        n[i] = norm(x[i])
    return n


X = gammaRW1D(100, 2, 1)
Y = list(range(1, 101))
plt.plot(Y, X)
plt.show()

# plt.savefig('something.png')
