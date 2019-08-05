import numpy as np
import matplotlib.pyplot as plt
from q3_posterior import q3_posterior
from checking3g import checking3g

checking3g()

# some values for mu
mu = np.linspace(0, 1, 201, True)

# case 1
fig = plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
m=1
H = 1
a = 10
Z = 1/923780
prob = q3_posterior(mu, m, H, a, Z)
plt.plot(mu, prob)
plt.title('m=1, H=1')
plt.xlabel(r'$\mu$')
plt.ylabel('posterior')

# case 2
plt.subplot(1,3,2)
m= 100
H = 100
a = 10
Z = 1/923780
prob = q3_posterior(mu, m, H, a, Z)
plt.plot(mu, prob)
plt.title('m=100, H=100')
plt.xlabel(r'$\mu$')
plt.ylabel('posterior')

# case 3
plt.subplot(1,3,3)
m=100
H = 80
a = 10
Z = 1/923780
prob = q3_posterior(mu, m, H, a, Z)
plt.plot(mu, prob)
plt.title('m=100, H=80')
plt.xlabel(r'$\mu$')
plt.ylabel('posterior')
plt.show()

fig.savefig('q3g.png', dpi = 300)

