import numpy as np
import matplotlib.pyplot as plt
from q3_likelihood import q3_likelihood
from checking3c import checking3c

#calling checking function
checking3c()

# some example values for mu
mu = np.linspace(0, 1, 201, True)

fig = plt.figure(figsize=(10,10))

# case 1
plt.subplot(1, 3, 1)
m = 1
H = 1
lik = q3_likelihood(mu, m, H)
plt.plot(mu, lik)
plt.title('m=1, H=1')
plt.xlabel(r'$\mu$')
plt.ylabel('L')

# case 2
plt.subplot(1, 3, 2);
m = 100;
H = 100;
lik = q3_likelihood(mu, m, H);
plt.plot(mu, lik);
plt.title('m=100, H=100');
plt.xlabel(r'$\mu$')
plt.ylabel('L');

# case 3
plt.subplot(1, 3, 3);
m = 100;
H = 80;
lik = q3_likelihood(mu, m, H);
plt.plot(mu, lik);
plt.title('m=100, H=80');
plt.xlabel(r'$\mu$')
plt.ylabel('L');
plt.show()

fig.savefig('q3c.png', dpi = 300)
