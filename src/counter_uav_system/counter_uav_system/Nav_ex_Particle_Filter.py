import numpy as np
import matplotlib.pyplot as plt

def diff_drive(x, u, omega, v, dT):
    x = x + dT * u * v
    u = u + dT * np.array([[0, -1], [1, 0]]) @ u * omega
    u = u / np.linalg.norm(u)  # renormalization of u
    return x, u

def draw_cov_ellipse(e, C, r):
    th = np.arange(0, 2 * np.pi, 0.1)
    I, H = np.linalg.eigh(C)
    X = H @ np.diag(np.sqrt(I)) @ np.array([np.cos(th), np.sin(th)]) * r
    plt.plot(e[0] + X[0, :], e[1] + X[1, :], '.')

def draw_ellipsoid(cx, cy, r1, r2):
    th = np.arange(0, 2 * np.pi, 0.01)
    x = np.array([cx + r1 * np.cos(th), cy + r2 * np.sin(th)])
    plt.plot(x[0, :], x[1, :], 'r.')

def fum(dT, hx, hy):
    arr = np.array([
        (dT * hx) / 2.0, (dT * hx) / 2.0,
        (dT * hy) / 2.0, (dT * hy) / 2.0,
        -dT * hy, dT * hy,
        dT * hx, -dT * hx
    ])
    return arr.reshape((4, 2))

def fxm(dT, wl, wr):

    arr = np.array([
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        dT * (wl / 2.0 + wr / 2.0), 0.0, 1.0, -dT * (wl - wr),
        0.0, dT * (wl / 2.0 + wr / 2.0), dT * (wl - wr), 1.0
    ])
    arr = arr.reshape((4, 4))
    return arr.transpose()

T = 10  # simulation time
dT = 0.2  # simulation time step
N = int(np.ceil(T / dT))
x = np.zeros((2, N))  # positions
u = np.zeros_like(x)  # headings
x_o = np.zeros((2, N))  # position estimate from odometry
u_o = np.zeros_like(x_o)  # headings estimate from odometry
u[:, 0] = [1, 0]  # initial heading
u_o[:, 0] = [1, 0]
Sx = np.zeros((4, 4))
su = 0.1  # wheel noise std
Su = np.diag([su ** 2, su ** 2])

# initial particles
NP = 100
P = np.vstack([np.tile(x_o[:, 0], (NP, 1)).T, np.tile(u_o[:, 0], (NP, 1)).T])
NNP = 50  # expansion ratio

for i in range(N - 1):
    wr_o = 1.0
    wl_o = 0.9
    v_o = (wr_o + wl_o) / 2
    omega_o = (wr_o - wl_o)
    X, U = diff_drive(x_o[:, i], u_o[:, i], omega_o, v_o, dT)
    x_o[:, i + 1] = X
    u_o[:, i + 1] = U

    wr = wr_o + np.random.randn() * su  # add noise to wheel speeds
    wl = wl_o + np.random.randn() * su
    v = (wr + wl) / 2
    omega = (wr - wl)
    X, U = diff_drive(x[:, i], u[:, i], omega, v, dT)
    x[:, i + 1] = X
    u[:, i + 1] = U
    Dm1 = np.linalg.norm(X)  # compute distance to [0,0]
    Dm2 = np.linalg.norm(X - np.array([10, 0]))  # compute distance to [10,0]

    Pex = np.empty((4, 0))
    for j in range(NP):
        ffu = fum(dT, P[2, j], P[3, j])
        X, U = diff_drive(P[0:2, j], P[2:4, j], omega_o, v_o, dT)
        noise = ffu @ (np.sqrt(Su) @ np.random.randn(2, NNP))
        temp=np.tile(np.hstack([X, U]).reshape(4, 1), (1, NNP))
        #print(np.hstack([X, U]).reshape(4, 1))
        expanded = np.tile(np.hstack([X, U]).reshape(4, 1), (1, NNP)) + noise
        Pex = np.hstack([Pex, expanded])

    Dp1 = np.sqrt(np.sum(Pex[0:2, :] ** 2, axis=0))
    Dp2 = np.sqrt(np.sum((Pex[0:2, :] - np.array([[10], [0]])) ** 2, axis=0))

    W1 = np.exp(-0.5 * (Dm1 - Dp1) ** 2 / 0.5 ** 2)
    W2 = np.exp(-0.5 * (Dm2 - Dp2) ** 2 / 0.5 ** 2)
    Js = np.argsort(W1)  # sort weights
    #Js = np.argsort(W1 * W2)  # alternative sorting
    
    Pex = Pex[:, Js]
    P = Pex[:, -NP:]

    ffx = fxm(dT, wl_o, wr_o)
    ffu = fum(dT, u_o[0, i], u_o[1, i])
    Sx = ffx @ Sx @ ffx.T + ffu @ Su @ ffu.T  # celebrated error propagation formula

    plt.clf()
    plt.plot(x[0, :i + 2], x[1, :i + 2], 'b.', label='x')
    plt.plot(x_o[0, :i + 2], x_o[1, :i + 2], '*b', label='x_o')
    plt.plot(Pex[0, :], Pex[1, :], 'g.', label='Pex')
    plt.plot(P[0, :], P[1, :], 'r.', label='P')
    draw_ellipsoid(0, 0, Dm1, Dm1)
    draw_ellipsoid(10, 0, Dm2, Dm2)
    plt.axis([0, 10, 0, 10])
    plt.pause(0.1)

plt.plot(x[0, :], x[1, :], label='x')
plt.plot(x_o[0, :], x_o[1, :], '*b', label='x_o')
Xm = np.zeros((2, 1))
Xm[:, 0] = x[:, i]

plt.plot(Xm[0, :], Xm[1, :], '.')
draw_cov_ellipse(x_o[:, i], Sx[0:2, 0:2], 2.0)
plt.plot(P[0, :], P[1, :], 'b.')
plt.show()

