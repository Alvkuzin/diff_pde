# diff_pde: snumerically solves a system of 1d nonlinear PDEs of diffusive type.

Currently it's just one function that you'll need. For $u = u(t, x)$, 
\begin{equation}
    \frac{\partial u}{\partial t} = \frac{\partial}{\partial x} \left( D \frac{\partial u}{\partial x} \right) + f,
\end{equation}
where 
\begin{equation}
    D = D(u, t, x), f = f(u, t, x).
\end{equation}

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


The package consists of six main classes, you can find their description and usage examples in ``tutorials``:
 
 1. ``Orbit`` (self-explanatory);
Let's initiate an Orbit object and plot an orbit: 
```python
from ibsen.orbit import Orbit
import matplotlib.pyplot as plt   
import numpy as np
DAY = 86400
orb = Orbit(T=25*DAY, e=0.7, M=30*2e33,
                 nu_los=90*np.pi/180, incl_los=20*np.pi/180)
t = np.linspace(-20*DAY, 70*DAY, 1000)
plt.plot(orb.x(t), orb.y(t))
```

## Details of the numerical method. 


## License 

This project is licensed under the MIT License. See the LICENSE file for details.
---


