import numpy as np
import matplotlib.pyplot as plt

N = 100

def identity_kernel(x, y):
    return x * y

def radial_basis_kernel(x, y, l):
    return np.exp(-(np.power(x - y, 2)/(2*np.power(l, 2))))

def rational_quadratic_kernel(x, y, a, l):
    return np.power(1. + np.power(x - y, 2) / (2 * a * l * l), -a)

xi, yi = np.mgrid[-(N/2):(N/2), -(N/2):(N/2)]

z = zip(xi.flatten(), yi.flatten())

print "Producing plot for identity kernel..."
identity_distances = []
for pair in z:
    identity_distances.append(identity_kernel(pair[0], pair[1]))

identity_distances = np.reshape(identity_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, identity_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Identity kernel")
plt.show()

print "Producing plot for radial basis kernel..."
radial_basis_distances = []
l = 2
for pair in z:
    radial_basis_distances.append(radial_basis_kernel(pair[0], pair[1], l))

radial_basis_distances = np.reshape(radial_basis_distances, [-1, xi.shape[1]])
print "radial basis distances = ", radial_basis_distances

plt.pcolormesh(xi, yi, radial_basis_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Radial Basis kernel (l = %s)" % l)
plt.show()

radial_basis_distances = []
l = 10
for pair in z:
    radial_basis_distances.append(radial_basis_kernel(pair[0], pair[1], l))

radial_basis_distances = np.reshape(radial_basis_distances, [-1, xi.shape[1]])
print "radial basis distances = ", radial_basis_distances

plt.pcolormesh(xi, yi, radial_basis_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Radial Basis kernel (l = %s)" % l)
plt.show()

radial_basis_distances = []
l = 25
for pair in z:
    radial_basis_distances.append(radial_basis_kernel(pair[0], pair[1], l))

radial_basis_distances = np.reshape(radial_basis_distances, [-1, xi.shape[1]])
print "radial basis distances = ", radial_basis_distances

plt.pcolormesh(xi, yi, radial_basis_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Radial Basis kernel (l = %s)" % l)
plt.show()

print "Producing plot for rational quadratic kernel..."
rational_quadratic_distances = []
a = 1
l = 1
for pair in z:
    rational_quadratic_distances.append(rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s)" % (a, l))
plt.show()

rational_quadratic_distances = []
a = 1
l = 5
for pair in z:
    rational_quadratic_distances.append(rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s)" % (a, l))
plt.show()

rational_quadratic_distances = []
a = 1
l = 10
for pair in z:
    rational_quadratic_distances.append(rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s)" % (a, l))
plt.show()

rational_quadratic_distances = []
a = 1
l = 15
for pair in z:
    rational_quadratic_distances.append(rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s)" % (a, l))
plt.show()

rational_quadratic_distances = []
a = 0.5
l = 15
for pair in z:
    rational_quadratic_distances.append(rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s)" % (a, l))
plt.show()

rational_quadratic_distances = []
a = 0.1
l = 15
for pair in z:
    rational_quadratic_distances.append(rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s)" % (a, l))
plt.show()

rational_quadratic_distances = []
nu = 2
a = 0.1
l = 15
for pair in z:
    rational_quadratic_distances.append(nu**2 * rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s, nu = %s)" % (a, l, nu))
plt.show()

rational_quadratic_distances = []
nu = 5
a = 0.1
l = 15
for pair in z:
    rational_quadratic_distances.append(nu**2 * rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s, nu = %s)" % (a, l, nu))
plt.show()

rational_quadratic_distances = []
nu = 1
a = 2
l = 2
for pair in z:
    rational_quadratic_distances.append(nu**2 * rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s, nu = %s)" % (a, l, nu))
plt.show()

rational_quadratic_distances = []
nu = 3
a = 2
l = 2
for pair in z:
    rational_quadratic_distances.append(nu**2 * rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s, nu = %s)" % (a, l, nu))
plt.show()

rational_quadratic_distances = []
nu = 6
a = 2
l = 2
for pair in z:
    rational_quadratic_distances.append(nu**2 * rational_quadratic_kernel(pair[0], pair[1], a, l))

rational_quadratic_distances = np.reshape(rational_quadratic_distances, [-1, xi.shape[1]])

plt.pcolormesh(xi, yi, rational_quadratic_distances, cmap=plt.cm.Greens_r)
plt.colorbar()
plt.title("Rational Quadratic kernel (a = %s, l = %s, nu = %s)" % (a, l, nu))
plt.show()
