import numpy
from matplotlib import pyplot, cm
import time


nx = 100
ny = 100
nt = 1
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)

rho = 1
nu = .1
dt = .001

u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))


def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                             (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

    return b


def pressure_poisson(p, dx, dy, b):
    pn = numpy.empty_like(p)
    pn = p.copy()

    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                         (2 * (dx ** 2 + dy ** 2)) -
                         dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                         b[1:-1, 1:-1])

        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
        p[-1, :] = 0  # p = 0 at y = 2

    return p


def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((ny, nx))
    btime = 0
    ptime = 0
    maintime = 0
    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)

        p = pressure_poisson(p, dx, dy, b)

        row, col = u.shape
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                u[i, j] = (un[i, j] -
                                 un[i, j] * dt / dx *
                                 (un[i, j] - un[i, j - 1]) -
                                 vn[i, j] * dt / dy *
                                 (un[i, j] - un[i - 1, j]) -
                                 dt / (2 * rho * dx) * (p[i, j + 1] - p[i, j - 1]) +
                                 nu * (dt / dx ** 2 *
                                       (un[i, j + 1] - 2 * un[i, j] + un[i, j - 1]) +
                                       dt / dy ** 2 *
                                       (un[i + 1, j] - 2 * un[i, j] + un[i - 1, j])))

                v[i, j] = (vn[i, j] -
                                 un[i, j] * dt / dx *
                                 (vn[i, j] - vn[i, j - 1]) -
                                 vn[i, j] * dt / dy *
                                 (vn[i, j] - vn[i - 1, j]) -
                                 dt / (2 * rho * dy) * (p[i + 1, j] - p[i - 1, j]) +
                                 nu * (dt / dx ** 2 *
                                       (vn[i, j + 1] - 2 * vn[i, j] + vn[i, j - 1]) +
                                       dt / dy ** 2 *
                                       (vn[i + 1, j] - 2 * vn[i, j] + vn[i - 1, j])))

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1  # set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0

    return u, v, p


if __name__ == '__main__':

    u = numpy.zeros((ny, nx))
    v = numpy.zeros((ny, nx))
    p = numpy.zeros((ny, nx))
    b = numpy.zeros((ny, nx))
    nt = 100

    start = time.time()

    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    end = time.time()

    print(end - start)

    fig = pyplot.figure(figsize=(11,7), dpi=100)
    # plotting the pressure field as a contour
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    pyplot.colorbar()
    # plotting the pressure field outlines
    pyplot.contour(X, Y, p, cmap=cm.viridis)
    # plotting velocity field
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.show()
