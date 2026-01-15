from diff_pde.lines_method import solve_diff_pde
import numpy as np

D = 3
T_init = 1
T_right = 0.5


def T0(x):
    x = np.asarray(x)
    return np.ones(x.size) * T_init

xi = 0.0
xf = 3.0
Nx=501
x_grid = np.linspace(xi, xf, Nx)

ti = 0
tf = 5
Nt = 303
t_ev = np.linspace(ti, tf, Nt)

x_ , [T_,] = solve_diff_pde(rhs_list = [lambda u, t, x: x*0],
                        D_list = [D],
                        x_grid=x_grid,
                        t_span=[ti, tf],
                        t_eval=t_ev,
                        u0_list=[T0],
                        bc_type=[[
                            "neumann", # on the left
                            "dirichlet" # on the right
                            ]
                                 ],
                        bound_conds = [[
                            lambda t: 0*t, # no-flux on the left
                            lambda t: 0*t +  T_right # constant T_right on the right
                            ]
                                 ], 
                        )

# Xtot, Ytot = trapezoid(X_, x_, axis=0), trapezoid(Y_, x_, axis=0)
frames = T_.T 
# frames_y = Y_.T 

from matplotlib.animation import FuncAnimation

from matplotlib import pyplot as plt
fig, ax = plt.subplots(ncols=1, figsize=(6, 5), constrained_layout=True)
# line_fix, = ax.plot(x_, frames[0], c='k', alpha=0.2)
line_fix_init, = ax.plot(x_, frames[0], c='k', alpha=0.2, ls='--',
                          label = 'Initial distribution of T')
line_fix_end, = ax.plot(x_, frames[-1], c='k', alpha=0.2,
                         label = 'Final distribution of T')


line, = ax.plot(x_, frames[0], c='r', label = 'Distribution of T')
# line_y, = ax.plot(x_, frames_y[0], c='b', label = 'Distribution of Y')
title = ax.set_title(f"t = {t_ev[0]:.3f}")

# l_t_x = ax1.plot(t_ev, Xtot, label = 'Integral X(t)', c='r', alpha=0.3)
# l_t_y = ax1.plot(t_ev, Ytot, label = 'Integral Y(t)', c='b', alpha=0.3)

# point_x = ax1.scatter(x_[0], Xtot[0], color='r')
# point_y = ax1.scatter(x_[0], Ytot[0], color='b')


# vmin = np.percentile(frames, 1)
# vmax = np.percentile(frames, 99)
# min_ = min(np.min(frames), np.min(frames_y))
# max_ = max(np.max(frames), np.max(frames_y))

# ax.set_ylim(min_ / 1.2, max_ * 1.2)
# ax.legend()
# ax1.legend()
ax.legend()

# every_nth_frame = 10

def update(k):
    line.set_data(x_, frames[k])
    # line_y.set_data(x_, frames_y[k])
    
    title.set_text(f"t = {t_ev[k]:.3f} (frame {k+1}/{len(t_ev)})")
    # point_x.set_offsets([[t_ev[k], Xtot[k]]])
    # point_y.set_offsets([[t_ev[k], Ytot[k]]])
    return line, title

nframes = len(t_ev) 
t_sec = 8.0  # desired total duration in seconds
interval_ms = 1000.0 * t_sec / nframes

anim = FuncAnimation(fig, update, frames=nframes, interval=interval_ms, blit=False)
plt.show()