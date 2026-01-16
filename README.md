# diff_pde: numerical solution of a system of 1d nonlinear PDEs of diffusive type.

One function that will give you the solution.

The purpose of this package is to solve one-dimensional partial differential equations of parabolic type (heat/diffusion type equations). The package can handle one equation or a system, linear and non-linear cases.

In the most general way, the problem is formulated like this. For the vector of unknown fields $\vec{u} = [u\_1(t, x), ..., u\_m(t, x)]$, 
on a spatial domain of $x = [x\_i, x\_f]$, 
```math
\begin{equation}
    \frac{\partial u_j}{\partial t} = D_\mathrm{out,~j}\frac{\partial}{\partial x} \left( D_j \frac{\partial u_j}{\partial x} \right) + f_j,\,\,j=1...m,
\end{equation}
```
where the diffusion coefficient, "outer" diffusion coefficient, and source function are all functions of the field, spatial, and temporal coordinates:
```math
\begin{equation}
    D_j = D_j(u, t, x), \,\,\,D_\mathrm{out,~j} = D_\mathrm{out,~j}(u, t, x), \,\,\,f_j = f_j(u, t, x).
\end{equation}
```

The equation must be supplemented with initial and boundary conditoins. The initial condition:
```math
\begin{equation}
    \vec{u}(t=0, x) = [u_{1, 0}(x), ..., u_{m, 0}(x)],
\end{equation}
```

and for boundary conditions there are three options. 
 - Periodic (only works for all fields at once, i.e. you
cannot set periodic BCs for $u\_1$ and Dirichlet BCs for $u\_2$):
```math
\begin{equation}
    \vec{u}(t, x=x_i) = \vec{u}(t, x=x_f),
\end{equation}
```
 - Dirichlet:
```math
\begin{equation}
    u_j(t, x_{i/f}) = g_{j, \mathrm{left/right}}(t),
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

You feed the source functions and diffusion coefficients, as well as initial/boundary conditions, and get the
solution on a grid of $\mathrm{spatial\~grid} \times \mathrm{time\~grid}$. 


## Installation & Requirements 


The solver is written using numpy and scipy. 

Normal installation:

```bash
git clone https://github.com/Alvkuzin/diff_pde
cd diff_pde
pip install .
```

For the installation and contributing, replace the last line with 

```bash
pip install -e .
```

## Usage

The main idea is very simple: to solve for $m$ fields $u_j$, you provide:
 - an m-element list of source functions $f(u, t, x)$,
 - an m-element list of coefficients D, each is either a scalar or a function $D(u, t, x)$,
 - optinally, an m-element list of coefficients $D\_\mathrm{out}$, each is either a scalar or a function $D\_\mathrm{out}(u, t, x)$,
while by default they are set to unity,
 - an m-element list of initial conditions: functions $u_0(x)$,
 - an m-element list of 2-element lists of boundary conditions: $[g_\mathrm{left}(t), g_\mathrm{right}(t)]$,
 - a spatial grid $x_\mathrm{grid}$ as a numpy array; it defines nods at which the fields will be defined as grid functions,
 - a time grid at which the fields are to be returned.

As a return, you get the m-element list of fields, each is a numpy array (x.size, t.size).

## Example 1. Heat equation.
Let's assume there's a homogenious metallic rod of length $L$ heated to a temperature $T_i(x)$. 
Starting from time $t=0$, the left edge of the rod is kept isolated ($\partial\_x T|\_{x=0}=0$),
while the right edge is kept at a temperature $T\_\mathrm{right}$. The temperature distribution
as a functtion of time $T(t, x)$ is described by the equation:
```math
\begin{split}
\partial_t T(t, x) = D \partial^2_x T(t, x),\\
\partial_x T|_{x=0} = 0,\\
T(t, L) = T\_\mathrm{right},\\
T(t=0, x) = T_i(x).
\end{split}
```

Here the coefficient $D$, the diffusion coefficient, is a property of the rod, and assumed to be a scalar. Note that
the final stationary temperature distribution is easily guessed: $T\_\mathrm{final}(x) \equiv T\_\mathrm{right}$. Here is the code 
for the numerical solution:
```python
from diff_pde import solve_diff_pde
import numpy as np

D = 3
L = 3.0
T_init = 1
T_right = 0.5

def T0(x):
    x = np.asarray(x)
    return np.ones(x.size) * T_init

xi = 0.0
xf = L
Nx=501
x_grid = np.linspace(xi, xf, Nx)

ti = 0
tf = 5
Nt = 303
t_ev = np.linspace(ti, tf, Nt)

x_ , [T,] = solve_diff_pde(rhs_list = [lambda u, t, x: x*0],
                        D_list = [D], # since D is a scalar here, it can be passed as `D_out_list=[D]` instead.
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
In the end, you have an array `T` of shape: (x\_.size, t\_ev.size).

You can run the code `simple_heat_anim.py` from terminal to see this solution animated.

## Example 2. Brusselator.
[The Brusselator equations](https://en.wikipedia.org/wiki/Brusselator) describe an autocatalyctic reaction
of two chemicals, which concentrations we shall denote $X(t, x)$ and $Y(t, x)$. Two chemicals
react with each other, as well as diffuse in space, which in one-dimentional case
results in the following reaction-diffusion system of equations:

```math
\begin{split}
\partial_t X = A - (B+1)X + X^2Y + \partial_x(D_x \partial_x X),\\
\partial_t Y = BX - X^2Y + \partial_x(D_y \partial_x Y),\\
+ \mathrm{some~initial~conditions}\\
+ \mathrm{some~boundary~conditions}
\end{split}
```
For boundary conditions, it makes sense to adopt either
periodic BCs (as if a small domain of a bigger picture is being simulated)
 or Neumann ones (absense of flux through the boundaries). For illustration purposes, let's
take diffusion coeffients $D\_x$ and $D\_y$ sligtly spatially-dependent and non-linear (u-dependent).

```python
from diff_pde import solve_diff_pde
import numpy as np

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
                        )
```
You can run the code `brusselator_anim.py` from terminal to see this solution animated.

## Example 3. Discontinuous diffusion coefficient.

Let us reformulate slightly the problem from the first example. Imagine two rods,
with lengths $L\_1, L\_2$, diffusion coeffients $D\_1, D\_2$, heated to temperatures
$T\_\mathrm{init, 1}, T\_\mathrm{init, 2}$. At the time $t=0$, one connects these two rods, 
while, as before, isolating the resulting rod of length $L\_1+L\_2$ on the left edge and keeping 
it at a constant temperature $T\_\mathrm{right}$ at the right edge. 

To find the
distribution of the temperature on this resulting rod, we would need to evolve 
$T\_1(t, x)$ and $T\_2(t, x)$ with different diffusion coefficients, keeping
$T\_1(t, L\_1) = T\_2(t, L\_1)$ and  $D\_1 \partial\_x T\_1(t, L\_1) = D\_2 \partial\_x  T\_2(t, L\_1)$.
Fortunately, since the code is written in a conservative manner, we can simply look for the solution
at the whole $x = [0, L\_1 + L\_2]$ domain with discontinuous diffusion coefficient:
```math
\begin{equation}
    D(x) = 
    \begin{cases}
        D1, \,\, 0 < x < L_1,\\
        D2, \,\, L_1 < x < L_2.\\
    \end{cases}
\end{equation}
```

```python
from diff_pde import solve_diff_pde
import numpy as np

D1 = 8
D2 = 0.5
L1 = 2.0
L2 = 1.0
T_init_1 = 1
T_init_2 = 2
T_right = 1.5

def T_init(x):
    x = np.asarray(x)
    return (T_init_1 * np.heaviside(-(x-L1), 0.5) +
            T_init_2 * np.heaviside((x-L1), 0.5)
                )

def D(u, t, x):
    x = np.asarray(x)
    return (D1 * np.heaviside(-(x-L1), 0.5) +
            D2 * np.heaviside((x-L1), 0.5)
                )

xi = 0.0
xf = L1 + L2
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
                        u0_list=[T_init],
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
You can run the code `discontin_heat_anim.py` from terminal to see this solution animated.


## Details of the numerical method. 
This code is written using the method of lines. It means that the desired solution $u(x)$ is approximated as a grid function
at a grid $x\_g: u(x\_i) = u\_i$, making a vector of $u\_i, i=1,...,N$. Then the right side of the equation is approximated as
```math
\begin{equation}
\partial_x (D \partial_x u) + f \approx L u_i + f(u_i, t, x_i),
\end{equation}
```
where $L$ is an operator acting on a vector $u\_i$, so that $L u\_i$ is an approximation of the diffusion part,

and the partial differential equation becomes the system of ODEs:
```math
\begin{equation}
\frac{\mathrm{d} u_i}{\mathrm{d} t} \approx L u_i + f(u_i, t, x_i).
\end{equation}
```
This system is solved using `scipy.integrate.solve_ivp`, which allows us to avoid thinking how 
is this system actually being solved. It is important to use `method='BDF'` (default) or `method='Radau'`,
or any other _implicit_ methods which for the equation $dU/dt=f$ not just make an eulerian step
$U\_{k+1} = U\_{k} + \Delta t f(U\_{k}, t_k)$ but solve the nonlinear equation
$U\_{k+1} = U\_{k} + \Delta t f(U\_{k+1}, t_{k+1})$ at each time step. This is the equivalent
of using the implicitly stable numerical scheme. 

### Note 1. System of PDEs.
For a system of equations for fields [$u\_1,...,u\_m$], the right-hand side is calculated as described above for each field and then concatenated into 
one vector, and the m
fields are also packed into a one big vector [$u\_0...u\_N,u\_{N+1},...,u\_{mN}$] for which the ODE is solved with the concatenated right-hand. 

### Note 2. Internal grid.
Internally, the solution is looked for at the spatial grid $\xi$ which in case of Dirichlet/Neumann BCs coincides with the provided
x-grid. In case of periodic boundary conditions, $\xi$ is an original x-grid with 
the rightmost point omitted. The rightmost point for the solutions $u\_j$ is then manually added as $u\_j(t, x\_f) \equiv u\_j(t, x\_i)$.
This is the numerical reason why periodic boundary conditions cannot be mixed with other BCs for different fields: they would "live" on 
different grids. This problem may be overcome, though.

### Note 3. Order of finite differences approximations.

Outsourcing the ODEs to `solve_ivp` allows not to think about time steps, as they are made automaticly by the solver,
 which returns a solution at any desired $t\_\mathrm{eval}$. The choice of a spatial grid, though, is the responsibility
of a user.

I attempted to write a code with the second-order spatial approximations. I tested it on some 'good' functions,
and the metric 
```math
\begin{equation}
R = \sqrt{\frac{1}{N\_x} \sum_{ij} (u^{ij}\_\mathrm{numerical} - u^{ij}\_\mathrm{analytical})^2 }
\end{equation}
```

 indeed behaves as
$R \propto 1/N\_x^2$, even for the non equally spaced grids. The test with the discontinuous diffusion coefficient, though, revealed $R \propto 1/N\_x$.
Note also that the arguments `rtol/atol` that may be provided to the `solve_diff_pde` refer to the arguments passed to the `solve_ivp`, e.g.
they do not guarantee these relative/absolute tolerance for a solution.

This project is licensed under the MIT License. See the LICENSE file for details.
---


