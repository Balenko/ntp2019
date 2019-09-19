import numpy
from matplotlib import pyplot, cm
import time
import multiprocessing
import os

nx = 100
ny = 100
nt = 1
nit = 100
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


def step_u(lx, ux, ly, uy,  un_step, vn_step, p_step, u_temp):
    ret = numpy.zeros_like(u_temp)
    for i in range(lx, ux):
        for j in range(ly, uy):
            ret[i, j] = (un_step[i, j] -
                         un_step[i, j] * dt / dx *
                         (un_step[i, j] - un_step[i, j - 1]) -
                         vn_step[i, j] * dt / dy *
                         (un_step[i, j] - un_step[i - 1, j]) -
                         dt / (2 * rho * dx) * (p_step[i, j + 1] - p_step[i, j - 1]) +
                         nu * (dt / dx ** 2 *
                         (un_step[i, j + 1] - 2 * un_step[i, j] + un_step[i, j - 1]) +
                          dt / dy ** 2 *
                          (un_step[i + 1, j] - 2 * un_step[i, j] + un_step[i - 1, j])))
    return ret


def step_v(lx, ux, ly, uy, un_step, vn_step, p_step, v_temp):
    ret = numpy.zeros_like(v_temp)
    for i in range(lx, ux):
        for j in range(ly, uy):
            ret[i, j] = (vn_step[i, j] -
                       un_step[i, j] * dt / dx *
                        (vn_step[i, j] - vn_step[i, j - 1]) -
                       vn_step[i, j] * dt / dy *
                       (vn_step[i, j] - vn_step[i - 1, j]) -
                       dt / (2 * rho * dy) * (p_step[i + 1, j] - p_step[i - 1, j]) +
                       nu * (dt / dx ** 2 *
                             (vn_step[i, j + 1] - 2 * vn_step[i, j] + vn_step[i, j - 1]) +
                             dt / dy ** 2 *
                             (vn_step[i + 1, j] - 2 * vn_step[i, j] + vn_step[i - 1, j])))
    return ret


def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((ny, nx))

    pool = multiprocessing.Pool(os.cpu_count())

    for n in range(nt):

        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        row, col = u.shape

        res_u = pool.starmap(step_u, [(1, round(row / 2), 1, round(col / 2), un, vn, p, u),
                                    (1, round(row / 2), round(col / 2), col - 1, un, vn, p, u),
                                    (round(row / 2), row - 1, 1, round(col / 2), un, vn, p, u),
                                    (round(row / 2), row - 1, round(col / 2), col - 1, un, vn, p, u)])

        u = numpy.zeros_like(u)
        for r in res_u:
            u += r

        res_v = pool.starmap(step_v, [(1, round(row / 2), 1, round(col / 2), un, vn, p, v),
                              (1, round(row / 2), round(col / 2), col - 1, un, vn, p, v),
                              (round(row / 2), row - 1, 1, round(col / 2), un, vn, p, v),
                              (round(row / 2), row - 1, round(col / 2), col - 1, un, vn, p, v)])

        v = numpy.zeros_like(v)
        for r in res_v:
            v += r

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

    p = numpy.zeros((ny, nx))
    b = numpy.zeros((ny, nx))
    nt = 100

    start = time.time()

    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    end = time.time()

    print(end - start)

    # x = ""
    # tokens = x.split("] [")
    # for x in range(100):
    #     token = (tokens[x][1:-1]).split(" ")
    #     print(token)
    #     for y in range(100):
    #         if token[y] != "":
    #             u[x, y] = float(token[y])
    #
    # x2 = ""
    # tokens2 = x2.split("] [")
    # for x in range(100):
    #     token2 = (tokens2[x][1:-1]).split(" ")
    #     for y in range(100):
    #         if token2[y] != "":
    #             v[x, y] = float(token2[y])


    fig = pyplot.figure(figsize=(11, 7), dpi=100)
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