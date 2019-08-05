import numpy as np
import matplotlib.pyplot as plt
from q3_prior import q3_prior
from checking3d import checking3d

#calling checking function
checking3d()

# some values for mu
mu = np.linspace(0, 1, 201, True)

# case 1
a = 2
Z = 1/6
prob = q3_prior(mu, a, Z)
fig = plt.figure(figsize=(10,10))
plt.plot(mu, prob, '-b')
plt.xlabel(r'$\mu$')
plt.ylabel('p(' + r'$\mu$' + ' a)')

# case 2
a = 10
Z = 1/923780
prob = q3_prior(mu, a, Z)
plt.plot(mu, prob, '-r')
plt.title('prior')
plt.legend(['a=2', 'a=10'])
plt.show()

fig.savefig('q3d.png', dpi = 300)

