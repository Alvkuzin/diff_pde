from diff_pde.lines_method import solve_diff_pde
import numpy as np
from scipy.integrate import trapezoid
from matplotlib.animation import FuncAnimation

from matplotlib import pyplot as plt

A = 2
B = 4
Dx0 = 1.6e-3
Dy0 = 8e-3


xi = 0.0
xf = 3.0
sgm = 0.1

Nx=501
ti = 0
tf = 150
Nt = int(3 * (tf - ti))
t_ev = np.linspace(ti, tf, Nt)

def fx(u, t, x):
    X_, Y_ = u
    return A - (B+1) * X_ + X_**2 * Y_ 

def fy(u, t, x):
    X_, Y_ = u
    return B * X_  - X_**2 * Y_ 

def x0(x):
    return A + 0.1 * np.cos(2 * np.pi * (x - xi) / (xf - xi) )**2

def y0(x):
    return B/A + 0.1 * np.sin(2 * np.pi * (x - xi) / (xf - xi) )**2

def Dx(u, t, x):
    X_, Y_ = u
    return (1+x)/(2+x) * Dx0 * (1 + 1/X_) / (1 + 1/Y_)

def Dy(u, t, x):
    X_, Y_ = u
    return (1.1+x)/(2.1+x) * Dy0  * (1.1 + 0.99/X_) / (1.01 + 0.98/Y_)


x_grid = np.linspace(xi, xf, Nx)
x_ , (X_, Y_) = solve_diff_pde(rhs_list = [fx, fy],
                        D_list = [Dx, Dy],
                        x_grid=x_grid,
                        t_span=[ti, tf],
                        t_eval=t_ev,
                        u0_list=[x0, y0],
                        rtol=1e-7,
                        bc_type="neumann"
                        # bc_type="dirichlet"
                        )

Xtot, Ytot = trapezoid(X_, x_, axis=0), trapezoid(Y_, x_, axis=0)
frames = X_.T 
frames_y = Y_.T 


fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(6, 5), constrained_layout=True)
# line_fix, = ax.plot(x_, frames[0], c='k', alpha=0.2)
line_fix_x, = ax.plot(x_, frames[-1], c='r', alpha=0.2, label = 'Final distribution of X')
line_fix_y, = ax.plot(x_, frames_y[-1], c='b', alpha=0.2, label = 'Final distribution of Y')


line, = ax.plot(x_, frames[0], c='r', label = 'Distribution of X')
line_y, = ax.plot(x_, frames_y[0], c='b', label = 'Distribution of Y')
title = ax.set_title(f"t = {t_ev[0]:.3f}")

l_t_x = ax1.plot(t_ev, Xtot, label = 'Integral X(t)', c='r', alpha=0.3)
l_t_y = ax1.plot(t_ev, Ytot, label = 'Integral Y(t)', c='b', alpha=0.3)

point_x = ax1.scatter(x_[0], Xtot[0], color='r')
point_y = ax1.scatter(x_[0], Ytot[0], color='b')


vmin = np.percentile(frames, 1)
vmax = np.percentile(frames, 99)
min_ = min(np.min(frames), np.min(frames_y))
max_ = max(np.max(frames), np.max(frames_y))

ax.set_ylim(min_ / 1.2, max_ * 1.2)
ax.legend()
ax1.legend()


every_nth_frame = 10

def update(k):
    line.set_data(x_, frames[k])
    line_y.set_data(x_, frames_y[k])
    
    title.set_text(f"t = {t_ev[k]:.3f} (frame {k+1}/{len(t_ev)})")
    point_x.set_offsets([[t_ev[k], Xtot[k]]])
    point_y.set_offsets([[t_ev[k], Ytot[k]]])
    return line, title

nframes = len(t_ev) 
t_sec = 8.0  # desired total duration in seconds
interval_ms = 1000.0 * t_sec / nframes

anim = FuncAnimation(fig, update, frames=nframes, interval=interval_ms, blit=False)
plt.show()