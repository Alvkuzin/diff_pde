import numpy as np
from scipy.integrate import solve_ivp


def _ddt(g, t, h=None):
    """Numerical derivative g'(t) for scalar function g(t)."""
    if h is None:
        h = 1e-8 * max(1.0, abs(t))
    return (g(t + h) - g(t - h)) / (2.0 * h)

def _as_D_callable(Dj, N: int):
    """
    Normalize diffusion coefficient specification to a callable returning shape (N,).

    Accepts:
      - scalar Dj: returns constant array
      - callable Dj(U, t, xi): must return array shape (N,)
    """
    if np.isscalar(Dj):
        Dj = float(Dj)
        return lambda U, t, xi: Dj * np.ones(N, dtype=float)
    if callable(Dj):
        return Dj
    raise TypeError("Each D in D_list must be a scalar or a callable.")

def _diffusion_conservative_1d_periodic_nonuniform(Dc, u, x_grid_full):
    """
    Calculates (D u_x)_x on a grid x_grid_full in case of the periodic 
    boundary conditions.
    
    x_grid_full: (N+1,) with x_grid_full[-1] = x_grid_full[0] + L
    Uses x = x_grid_full[:-1] as unique nodes.
    """
    x_full = np.asarray(x_grid_full, dtype=float)
    if x_full.ndim != 1 or x_full.size < 4:
        raise ValueError("x_grid_full must be 1D with length >= 4 (N+1, N>=3).")

    x = x_full[:-1]
    N = x.size
    if Dc.shape != (N,) or u.shape != (N,):
        raise ValueError(f"""Dc and u must have shape (N,) where N=len(x_grid_full)-1={N}
                         but have shapes: Dc: {Dc.shape}, u: {u.shape}""")

    # Check periodic endpoint
    L = x_full[-1] - x_full[0]
    if not (L > 0):
        raise ValueError("Need x_grid_full[-1] > x_grid_full[0] (positive period length).")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_grid_full[:-1] must be strictly increasing.")

    # Face spacings Δx_{i+1/2}
    dx_face = np.empty(N, dtype=float)
    dx_face[:-1] = x[1:] - x[:-1]
    dx_face[-1] = (x_full[-1] - x[-1])  # from last node to periodic endpoint (= first+L)

    # Cell widths Δx_i = x_{i+1/2} - x_{i-1/2}
    # with x_{i+1/2} = x_i + 0.5*dx_face[i]
    dx_cell = 0.5 * (dx_face + np.roll(dx_face, 1))

    # Face diffusion (harmonic mean)
    Dc_right = np.roll(Dc, -1)
    D_face = 2.0 * Dc * Dc_right / (Dc + Dc_right + 1e-300)

    # Fluxes on faces i+1/2
    u_right = np.roll(u, -1)
    F = D_face * (u_right - u) / dx_face

    # Divergence
    return (F - np.roll(F, 1)) / dx_cell

def _unpack_boudaries_for_diff(bc_type, g_func=None):
    """
    REDUNDANT. SHOULD REMOVE.
    
    Takes bc_type: should bee either a string or a list of 2 strings, and
    g_gunc: None, func, or list of two funcs. 
    
    Returns 
        bc (string: either "periodic" or "mixed"), 
        bcL, bcR (each is either bc_type, if it's a string, or unpacked 
                  bc_type, if it's a list)
        gL, gR (each is either g_func, if it's a string, or unpacked 
                  g_func, if it's a list. If g_func is None, each is a function
                  returning 0).        
    """

    if isinstance(bc_type, str):
        bc = bc_type.lower()
        bcL, bcR = bc, bc 
        gL, gR = g_func#, g_func
            
    elif (isinstance(bc_type, list) and len(bc_type)==2):
        bc = "mixed"
        bcL, bcR = bc_type
        bcL, bcR = bcL.lower(), bcR.lower()
        if bcL=='periodic' or bcR=='periodic':
            raise ValueError("periodic BC cannot be set only on one end of the domain.")
        gL, gR = g_func
        
    else:
        raise ValueError("""
                         bc_type should be either a string or a list 
                         of two strings.
                         """)
    return bc, bcL, bcR, gL, gR

def _diffusion_conservative_1d(Dc, u, x_grid, bc_type, g_func=None, 
                               if_periodic=False):
    """
    Compute (D u_x)_x with 2nd-order flux form on a uniform 1D grid.

    bc_type meanings:
      - "periodic": wrap
      - "neumann": prescribed gradients u_x at boundaries: u_x(left)=gL_t, u_x(right)=gR_t
      - "dirichlet": endpoints handled externally (we return 0 at endpoints)
      
     bc_type may be either a string: bc_type="neumann", etc... OR a list:
     bc_type = ['dirichlet', 'neumann']. Note that periodic BCs can only be 
     passed as one string (it doesn't make sense to pass e.g. 
                           ['periodic', 'dirichlet']).
    """
    # N = u.size
    if if_periodic:
        return _diffusion_conservative_1d_periodic_nonuniform(Dc, u,
                                                    x_grid_full=x_grid)
    
    bc, bcL, bcR, gL_t, gR_t = _unpack_boudaries_for_diff(bc_type, g_func)
    

    x = x_grid
    dx_face = x[1:] - x[:-1]        # Δx_{i+1/2}, shape (N-1,)
    
    # control-volume widths around nodes (2nd order FV style)
    dx_cell = np.empty_like(u, dtype=float)         # Δx_i
    dx_cell[1:-1] = 0.5 * (dx_face[1:] + dx_face[:-1])
    # dx_cell[0] = 0.5 * dx_face[0]
    # dx_cell[-1] = 0.5 * dx_face[-1]
    dx_cell[0] = dx_face[0]
    dx_cell[-1] = dx_face[-1]
    
    # if Dirichlet, enforce boundary values *for computing fluxes* (do NOT rely on old u[0])
    if bcL == "dirichlet":
        u = u.copy()
        u[0] = gL_t
    if bcR == "dirichlet":
        u = u.copy() if bcL != "dirichlet" else u
        u[-1] = gR_t
    
    # face diffusion coefficient (harmonic mean)
    a = Dc[:-1]
    b = Dc[1:]
    D_face = 2.0 * a * b / (a + b + 1e-300)        # shape (N-1,)
    
    # interior face fluxes: F[i] = F_{i+1/2}
    F = D_face * (u[1:] - u[:-1]) / dx_face        # shape (N-1,)
    
    out = np.zeros_like(u)
    
    # interior divergence is 2nd order on nonuniform grids
    out[1:-1] = (F[1:] - F[:-1]) / dx_cell[1:-1]
    
    # ---- boundary closures (2nd order) ----
    # We want fluxes at the boundary faces F_{-1/2} and F_{N-1/2}.
    
    # LEFT boundary face
    if bcL == "neumann":
        # print("left n")
        if gL_t is None:
            raise ValueError("Need gL_t for left Neumann BC (u_x at left).")
        # 2nd-order ghost point with mirrored spacing: x_{-1} = x0 - (x1-x0)
        # centered gradient at x0: (u1 - u_{-1}) / (2*dx0) = gL  => u_{-1} = u1 - 2*dx0*gL
        dx0 = dx_face[0]
        u_m1 = u[1] - 2.0 * dx0 * gL_t
        # flux at boundary face (between ghost and node 0): (u0 - u_{-1})/dx0
        F_left = Dc[0] * (u[0] - u_m1) / dx0
        out[0] = (F[0] - F_left) / dx_cell[0]
    elif bcL == "dirichlet":
        # print("left d")
        # Dirichlet: easiest & cleanest is to not evolve boundary node; keep diffusion RHS zero there.
        out[0] = 0.0
    else:
        raise ValueError("left boundary condition must be 'dirichlet' or 'neumann'.")
    
    # RIGHT boundary face
    if bcR == "neumann":
        # print("right n")
        if gR_t is None:
            raise ValueError("Need gR_t for right Neumann BC (u_x at right).")
        dxN = dx_face[-1]
        # ghost point mirrored: x_N = x_{N-1} + (x_{N-1}-x_{N-2})
        # centered gradient at x_{N-1}: (u_N - u_{N-2}) / (2*dxN) = gR => u_N = u_{N-2} + 2*dxN*gR
        u_pN = u[-2] + 2.0 * dxN * gR_t
        F_right = Dc[-1] * (u_pN - u[-1]) / dxN
        out[-1] = (F_right - F[-1]) / dx_cell[-1]
    elif bcR == "dirichlet":
        # print("right d")
        out[-1] = 0.0
    else:
        raise ValueError("left boundary condition must be 'dirichlet' or 'neumann'.")
    return out

            
def check_bcs(bc_type, regular_strs, special_strs, m):
    """
    Checks boundary conditions.
    Define allowed_strs as concatenation of regular_ and special_strs. Then:
    bc_type must be one of:
                      - a string in `allowed_strs`,
                      - or a list of 
                       -- strings, one of 'regular_strs'',
                       -- 2-element lists, each element one of 'regular_strs'.
                       
    m (int) is a lenth of a solution vector.
    
    Returns: (if_special (bool, if bc_type is a string in `special_strs`),
            bc_touse (list of 2-element lists.)
            )
    """
    allowed_strs = regular_strs + special_strs
    bc_touse = [[]] * m
    is_special = False
    if isinstance(bc_type, str):
        bc = bc_type.strip().lower()
        if bc not in allowed_strs:
            raise ValueError(f"""If bc_type is a string, it must be  one of
                             {allowed_strs}.""")
        if bc not in special_strs:
            bc_touse = [
                [bc, bc] for _im in range(m)
                ]
        else:
            is_special = True
    elif isinstance(bc_type, list):
        for j in range(m):
            bc_j = bc_type[j]
            if isinstance(bc_j, str):
                bc = bc_j.strip().lower()
                if bc not in regular_strs:
                    raise ValueError(f"""If bc_j is a string, it must be one of
                                     {regular_strs}.""")
                bc_touse[j] = [bc_j, bc_j]
            elif isinstance(bc_j, list):
                if len(bc_j) != 2 or not all([bc_j[j] in regular_strs for j in range(len(bc_j))]):
                    raise ValueError(f"""If bc_j is a list, it must be a 
                                     2-element list consisting of one of
                                     {regular_strs}.""")
                bc_touse[j] = bc_j
            else:
                raise ValueError(f"""All elements of the bc list should be
                                 either strings of {regular_strs}, or a 
                                 2-element lists, of {regular_strs}.""")
            
    else:
        raise ValueError(f"""bc_type must be one of:
                          - a string in {allowed_strs},
                          - or a list of 
                           -- strings, one of {regular_strs},
                           -- 2-element lists, each element one of {regular_strs}.
                        """)
    return is_special, bc_touse

def solve_diff_pde(
    rhs_list,
    D_list,
    *,
    x_grid,
    t_span: tuple[float, float],
    u0_list,
    bc_type: str = "periodic",
    bound_conds=None,          # [[gL0,gR0],...,[gL_{m-1},gR_{m-1}]]; functions of t
    t_eval=None,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    rhs_args_list=None,
):
    """
    Method-of-lines solver for m coupled 1D reaction–diffusion fields u_j(x,t):

        du_j/dt = d/dx( D_j(U,t,x) * du_j/dx ) + rhs_j(U, t, xi, *args_j),
        + u_j(t=t_i) = u0_l(x)
        + left/right boundary codition.

    Inputs
    ------
    rhs_list: list of m callables
        rhs_list[j](U_fields, t, xi, *args_j) -> array (N,)
        where U_fields is a list of length m, each element shape (N,).
        
    D_list: list of m items
        Each is either:
          - scalar diffusion coefficient, or
          - callable D_j(U_fields, t, xi, j) -> array (N,)
          
    x_grid: np.array of shape (N,)
        A grid to calculate the function on. Should be ascending.
        
    u0_list: list of m callables
        Initial conditions: u0_list[j](xi) -> array (N,)
        each element shape (N,).
    
    t_span: tuple (t_init, t_end).
        Times of start (initial condition is assumed to be at t_init) and 
        finish. 
        
    t_eval: np.array. 
        A time grid to return the solution at.      
    
    bc_type: "periodic" | "neumann" | "dirichlet". Either a string, 
        or an m-element list of
        strings:
            ['dirichlet', ..., 'neumann'],
            , or an m-element list of 1- and 2-element lists of strings:
            [
                ['d..', 'n..'], 'd..', ..., ['n..', 'd..']
                ]
        Note that 'periodic' may be currently only applied to the whole system,
        so you cannot pass e.g. bc_type=['d..', 'p..'], but only bc_type='p..'.
        
    bound_conds:
            List of boundary conditions functoins: f(t), with the same 
            structure as bc_type.
            If None, all functions are set to == 0
        If bc_type_j is "periodic", then bound_conds_j is ignored.
        
    method (default 'BDF'), rtol (default 1e-6), atol (default 1e-9) ---
    arguments passed to solve_ivp.

    BC conventions
    --------------
    - Dirichlet: u_j(left,t)=gLj(t), u_j(right,t)=gRj(t)
      Implemented by enforcing boundary values inside RHS evaluation and setting
      du/dt at boundaries to g'(t).
    - Neumann: u_{j,x}(left,t)=gLj(t), u_{j,x}(right,t)=gRj(t)
      Implemented via boundary fluxes in conservative diffusion operator.

    Returns
    -------
    xi : ndarray (N,)
    sol : OdeResult with y packed as [u0_block, u1_block, ..., u_{m-1}_block]
    """
    m = len(rhs_list)
    if rhs_args_list is None:
        rhs_args_list = [()] * m
    N_x_grid = x_grid.size
    if m != len(D_list):
        raise ValueError("rhs_list and D_list must have the same length.")
    if m != len(u0_list):
        raise ValueError("rhs_list and u0_list must have the same length.")
    if len(rhs_args_list) != m:
        raise ValueError("rhs_args_list must match rhs_list length.")

    if m < 1:
        raise ValueError("Need at least one field.")

    if N_x_grid < 3:
        raise ValueError("Require N >= 3.")
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("Require tf > t0.")
        
    if bound_conds is None:
        f_dummy = lambda _t: _t * 0
        bound_conds = [
                [f_dummy, f_dummy] for _im in range(m)
            ]
    elif isinstance(bound_conds, list):
        if not all([ len(bound_conds[j])==2 for j in range(len(bound_conds))]):
            raise ValueError("bound_conds should be either None, or a m-element list of 2-element lists. ")
    else:
        raise ValueError("bound_conds should be eitehr None, or a m-element list of 2-element lists. ")
        
    if_periodic, bc_to_use = check_bcs(bc_type = bc_type,
                                       regular_strs = ['dirichlet', 'neumann'],
                                       special_strs = ['periodic'],
                                       m = m)
    
    if if_periodic:
        xi = x_grid[:-1]
        # N = xi.size
    else:
        xi = x_grid
    N = xi.size
    # print(if_periodic)
    # print(N)
    # Normalize diffusion coefficients to callables
    D_callables = [_as_D_callable(Dj, N) for Dj in D_list]

    # Initialize fields
    def _init(ic):
        if callable(ic):
            v = np.asarray(ic(xi), dtype=float)
        else:
            v = np.asarray(ic, dtype=float)
        if v.shape != (N,):
            raise ValueError(f"IC must have shape (N,), got {v.shape}")
        return v

    U0_fields = [_init(ic) for ic in u0_list]
    if not if_periodic:
        for j in range(m):
            bcL, bcR = bc_to_use[j]
            gLj, gRj = bound_conds[j]
            if bcL == 'dirichlet':
                U0_fields[j][0] = float(gLj(t0))
            if bcR == 'dirichlet':
                U0_fields[j][-1] = float(gRj(t0))        
                
    U0 = np.concatenate(U0_fields)

    def unpack(U_flat):
        return [U_flat[j * N:(j + 1) * N] for j in range(m)]

    def pack(dU_fields):
        return np.concatenate(dU_fields)

    def rhs(t, U_flat):
        U_fields = unpack(U_flat)

        U_eff = []
        for j in range(m):
            uj = U_fields[j].copy()
            if not if_periodic:
                bcL, bcR = bc_to_use[j]
                gLj, gRj = bound_conds[j]
                if bcL == 'dirichlet':
                    uj[0] = float(gLj(t))
                if bcR == 'dirichlet':
                    uj[-1] = float(gRj(t))   
            U_eff.append(uj)

        dU = []

        for j in range(m):
            uj = U_eff[j]

            # Evaluate diffusion coefficient on centers
            Dc = np.asarray(D_callables[j](U_eff, t, xi), dtype=float)
            if Dc.shape != (N,):
                raise ValueError(f"D[{j}] must return shape (N,), got {Dc.shape}")

            # dirichlet values and Neumann gradients if needed

            bcL_, bcR_ = bound_conds[j]
            diff = _diffusion_conservative_1d(Dc, uj, x_grid, bc_to_use[j],
                                              g_func=[bcL_(t), bcR_(t)],
                                              if_periodic=if_periodic)

            # Reaction/source
            rj = np.asarray(rhs_list[j](U_eff, t, xi, *rhs_args_list[j]), dtype=float)
            if rj.shape != (N,):
                raise ValueError(f"rhs[{j}] must return shape (N,), got {rj.shape}")

            duj = diff + rj

            # Strong Dirichlet enforcement: du/dt at boundaries equals g'(t)
            if not if_periodic:
                bcL, bcR = bc_to_use[j]
                gLj, gRj = bound_conds[j]
                if bcL == "dirichlet":
                    # gLj, gRj = bound_conds[j]
                    duj[0] = _ddt(gLj, t)
                if bcR == 'dirichlet':
                    duj[-1] = _ddt(gRj, t)

            dU.append(duj)

        return pack(dU)

    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=U0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    U = unpack(sol.y)
    if if_periodic:
        U_with_endpoint = []
        for u_j in U:
            u_j_with_endpoint = np.zeros((x_grid.size, t_eval.size))
            u_j_with_endpoint[0:-1, :] = u_j
            u_j_with_endpoint[-1, :] = u_j[0, :]
            U_with_endpoint.append(u_j_with_endpoint)
            # u_j = np.concatenate([u_j, u_j[:1]])
        return x_grid, U_with_endpoint
    return xi, U
