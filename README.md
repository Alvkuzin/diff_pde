# diff_pde: snumerically solves a system of 1d nonlinear PDEs of diffusive type.

Currently it's just one function that you'll need. For $\vec{u} = [u\_1(t, x), ..., u\_m(t, x)]$, on a spatial domain of $x = [x\_i, x\_f]$, 
```math
\begin{equation}
    \frac{\partial u_j}{\partial t} = \frac{\partial}{\partial x} \left( D_j \frac{\partial u_j}{\partial x} \right) + f_j,\,\,j=1...m,
\end{equation}
```
where 
```math
\begin{equation}
    D_j = D_j(u, t, x), f_j = f_j(u, t, x).
\end{equation}
```
The initial conditions are set as:
```math
\begin{equation}
    \vec{u}(t=0, x) = [u_{1, 0}(x), ..., u_{m, 0}(x)],
\end{equation}
```
and for boundary conditions there are three options. 
 - Periodic (only works for all fields at once, you
cannot set periodic BCs for $u\_1$ and Dirichlet BCs for $u\_2$):
```math
\begin{equation}
    \vec{u}(t, x=x_i) = \vec{u}(t, x=x_f),
\end{equation}
```
 - Dirichlet:
```math
\begin{equation}
    u_j(t, x_{i\f}) = g_{j, \mathrm{left/right}}(t),
\end{equation}
```

 - Neumann:
```math
\begin{equation}
    \frac{ \partial u_j }{\partial x}|_{x_{i/f}} = g_{j, \mathrm{left/right}}(t).
\end{equation}
```

It is possible to apply Dirichlet and Neumann conditions in a mixed way, e.g. different BCs for 
different fields $u\_j$ on $x\_i/x\_f$. 

You feed the right-hand sides, as well as initial/boundary conditions, and get the
solution on a grid of spatial grid x time grid. 


## Installation & Requirements 


The solver is written using numpy and scipy. 

Normal installation:

```bash
git clone https://github.com/Alvkuzin/diff_pde
cd IBSEn
pip install .
```

For the installation and contributing, replace the last line with 

```bash
pip install -e .
```

## Usage

The main idea is very simple: to solve for $m$ fields $u_j$, you provide:
 - an m-element list of right-hand side functions $f(u, t, x)$,
 - an m-element list of coefficients D, each is either a scalar or a function $D(u, t, x)$,
 - an m-element list of initial conditions: functions $u_0(x)$,
 - an m-element list of 2-element lists of boundary conditions: $[g_\mathrm{left}(t), g_\mathrm{right}(t)]$.
 - a spatial grid $x_\mathrm{grid}$ as at numpy array; it defines nods at which the fields will be defined as grid functions.
 - a time grid at which the fields are to be returned.

As a return, you get the m-element list of fields, each is a numpy array (x.size, t.size).

## Example 1. Heat equation.
Let's assume there's a homogenious metallic rod of length $L$ heated to a temperature $T(x)$. 
Starting from time $t=0$, the left edge of the rod is kept isolated ($\partial\_x T|\_{x=0}=0$),
while the right edge is kept at a temperature $T\_0$. This problem is described by the equation:
```math
\begin{split}
\partial_t T(t, x) = D \partial^2_x T(t, x),\\
\partial_x T|_{x=x_i} = 0,\\
T(t, x_f) = T_0,\\
T(t=0, x) = T(x).
\end{split}
```

Here the coefficient $D$, the diffusion coefficient, is a property of the rod, and assumed to be a scalar. Note that
the final stationary temperature distribution is easily guessed: $T\_\mathrm{final}(x) \equiv T\_0$. Here is the code 
for the numerical solution:
```python
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
```
In the end, you have an array $T_$ of shape: (x\_.size, t\_ev.size).

Run the code `simple_heat_anim.py` to see this solution animated.

## Example 2. Brusselator.
[The Brusselator equations](https://en.wikipedia.org/wiki/Brusselator) describe an autocatalyctic reaction: 


```python
import numpy as np
from diff_pde.lines_method import solve_diff_pde

def f1(U, t, x):
    u1, u2 = U
    return ...

def f2(U, t, x):
    u1, u2 = U
    return ...



```



## Details of the numerical method. 


## License 

This project is licensed under the MIT License. See the LICENSE file for details.
---


