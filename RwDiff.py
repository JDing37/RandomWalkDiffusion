import random
import matplotlib as plt
from pylab import *

class RandomWalk:

    def one_D(self, N):
        u = 2*randint(0,2,size=N)-1
        x = cumsum(u)
        return x



rand = RandomWalk()
x = rand.one_D(1000)
N = list(range(1,1001))
plt.plot(x, N)
plt.savefig('x(N).png')
        
