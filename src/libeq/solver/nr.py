from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt

from libeq.excepts import FailedCalculateConcentrations, TooManyIterations
from libeq.utils import species_concentration

from .damping import pcf


def newton_raphson(
    x0: npt.NDArray,
    *,
    log_beta: npt.NDArray,
    log_ks: npt.NDArray,
    stoichiometry: npt.NDArray,
    solid_stoichiometry: npt.NDArray,
    total_concentration: npt.NDArray,
    solids_idx: List[int] = [],
    max_iterations: int = 1000,
    threshold: float = 1e-10,
    damping: bool = False,
    forcer: bool = False,
    scaling: bool = False,
    step_limiter: bool = True,
    zero_offdiag: bool = False,
    debug: bool = False,
    panic: bool = True,
    **kwargs: Dict[str, Any],
) -> npt.NDArray:
    r"""
    Solve the set of equations $J \cdot \delta c = -F$ using Newton-Raphson's method.

    Given an initial guess **x0** and the parameters of the titration, this
    function uses the Newton-Raphson method to solve the free
    concentrations.  It includes several
    convergence tecniques such as scaling, damping and step limiting. For the
    original source of damping and scaling see De Robertis, *et al.*,
    [^1] and also Del Piero, *et al.* [^2].


    [^1]: De Robertis, *et al.*: *Analytica Chimica Acta* 1986, **191**, 385-398
    [^2]: Del Piero, *et al.*: *Annali di Chimica* 2006, 96

    !!! warning

        This function is the very core of the program. Do not change
        the default parameters unless you know what you are doing.

    Parameters
    ----------
    x0 : numpy.ndarray
        The concentration array of shape (n, c+p), where n is the number of points c is the number of components and p is the number of precipitates.
    log_beta : numpy.ndarray
        The logarithm of the equilibrium constants with shape (n, s), where s is the number of soluble species.
    log_ks : numpy.ndarray
        The logarithm of the solubility products with shape (n, p), where p is the number of solid species.
    stoichiometry : numpy.ndarray
        The stoichiometry matrix with shape (n, s), where s is the number of soluble species.
    solid_stoichiometry : numpy.ndarray
        The stoichiometry matrix with shape (n, p), where s is the number of precipitable species.
    total_concentration : numpy.ndarray
        The total concentration vector with shape (n, c), where n is the number of points c is the number of components..
    solids_idx : List[int]
        Indices of solid species to compute.
    max_iterations : int
        Maximum number of iterations.
    threshold : float
        Convergence threshold.
    damping : bool
        Whether to apply damping.
    forcer : bool
        Whether to use a line search algorithm.
    scaling : bool
        Whether to apply scaling to the Jacobian matrix.
    step_limiter : bool
        Whether to limit the step size.
    zero_offdiag : bool
        Whether to zero out off-diagonal elements of the Jacobian matrix.
    debug : bool
        Whether to print debug information.
    panic : bool
        Whether to save intermediate results in case of failure.
    **kwargs : Dict[str, Any]
        Additional keyword arguments.

    Returns
    -------
    x : numpy.ndarray
        Array of shape (n, c+p) of final concentrations of all components in the system.

    Raises
    ------
    ValueError
        If the input is incorrect.
    FailedCalculateConcentrations
        If the Jacobian or residuals cannot be calculated.
    TooManyIterations
        If too many iterations are performed without convergence.
    """

    def _panic_save():
        if panic:
            np.savez_compressed(
                "consol_panic.npz",
                free_concentration=x0,
                log_beta=log_beta,
                stoichiometry=stoichiometry,
                analytc=total_concentration,
            )

    if zero_offdiag and scaling:
        raise ValueError(
            "Options scaling and zero_offdiag are not compatible with each other"
        )

    if "do_iterations" in kwargs:
        do_iterations = kwargs["do_iterations"]
        if not isinstance(do_iterations, int):
            raise TypeError("do_iteration must be a positive int.")
        if do_iterations < 1:
            raise ValueError("do_iteration must be a positive int.")
    else:
        do_iterations = None

    n_components = stoichiometry.shape[0]
    n_species = stoichiometry.shape[1]
    n_solids = solid_stoichiometry.shape[1]

    solids_to_remove = np.array(
        list(set(range(n_components, n_components + n_solids)) - set(solids_idx)),
        dtype=int,
    )
    # copy x0 so that it is not modified outside this function.
    x = np.delete(np.copy(x0), solids_to_remove, axis=-1)
    solid_stoichiometry = np.delete(
        np.copy(solid_stoichiometry), solids_to_remove - n_components, axis=-1
    )
    log_ks = np.delete(np.copy(log_ks), solids_to_remove - n_components, axis=-1)

    # ------ check input --------
    # libaux.assert_array_dim(2, x0)
    x, total_concentration = np.atleast_2d(x, total_concentration)

    # ------ main loop ----------
    for iterations in range(max_iterations):
        c0 = species_concentration(x, log_beta, stoichiometry, full=True)

        _c = c0

        F = fobj(
            _c,
            log_ks,
            stoichiometry,
            solid_stoichiometry,
            total_concentration,
        )

        J = jacobian(_c, stoichiometry, solid_stoichiometry)

        if np.any(np.isnan(J)):
            _panic_save()
            msg2 = f"could not calculate jacobian (iteration {iterations + 1})"
            raise FailedCalculateConcentrations(msg2, x)
        if np.any(np.isnan(F)):
            msg2 = f"could not calculate residuals (iteration {iterations + 1})"
            raise FailedCalculateConcentrations(msg2, x)

        # FIXME This should be deleted when debug is not necessary
        # if not iterations % 50:
        #     print('chisq(it:', iterations, 'n:', T.shape[0], ') = ',
        #           np.sum(F**2), np.max(np.abs(F)))

        if zero_offdiag:
            J *= np.eye(n_species)  # zerom

        if scaling:
            d = DRScaling(J, F)
            dx = np.linalg.solve(J, -F) / np.sqrt(d)
        else:
            dx = np.linalg.solve(J, -F)

        if forcer:
            step_length, _ = linesearch3(
                x, dx, log_beta, stoichiometry, total_concentration
            )
            x += step_length[:, None] * dx
        elif step_limiter:
            x[:, :n_components] += (
                limit_step(x[:, :n_components], dx[:, :n_components])
                * x[:, :n_components]
            )
            x[:, n_components:] += dx[:, n_components:]
        else:
            x += dx * x

        if (do_iterations and iterations + 1 >= do_iterations) or np.all(
            np.abs(F) < threshold
        ):
            if solids_to_remove.size != 0:
                temp = np.zeros((x.shape[0], n_components + n_solids))
                temp[:, ~np.isin(range(n_components + n_solids), solids_to_remove)] = x
                x = temp
            return x

        if damping:
            x = pcf(x, log_beta, stoichiometry, total_concentration)

    raise TooManyIterations("too many iterations", x)


def linesearch3(
    x0: npt.NDArray,
    dx: npt.NDArray,
    log_beta: npt.NDArray,
    stoichiometry: npt.NDArray,
    T: npt.NDArray,
    lmax=None,
    g0=None,
    g2=None,
):
    r"""Three-point parabolic line search.

    This functions implements a 3-point linesearch in the Newton direction.
    This is a variation of the line search for 2 points[^1]. The function to
    be minimized is the same though but the approach is different and it is
    adapted to the nature of the problem of concentration solving. We define
    a function $(f=\frac12F\cdot F)$ which is to be minimized. Then we
    define a parameter $\lambda$ that $0<\lambda<1$ which is the fraction of the Newton step
    and then another function *g* which is function of the fractional is
    defined so that:
    $$
    g(\lambda) = f(x_0 + \lambda \delta x)
    $$
    We know that negative $x_0$ values are forbidden, therefore $\lambda$ might limited
    to values lower than 1. The maximum value allowed for $\lambda$ is that that makes
    any concentration equal to 0, therefore
    $(\lambda_{max} = -x_0/\delta$ if $-x_0/\delta<1)$

    $g(\lambda)$ is modeled as a parabolic function for which we calculate
    the values for $\lambda=0$, $\lambda= \frac{1}{2}\lambda_{max}$ and $\lambda=0.99\lambda_{max}$.

    [^1]: W. H. Press, S. A. Teukolksy, W. T. Vetterling, Brian P. Flannery,
       Numerical Recipes in C. The Art of Scientific Computing, Second Edition
       1997, pages 384--385.

    Parameters
    ----------
    x0 : np.ndarray
        The initial state.
    dx : np.ndarray
        The Newton direction.
    log_beta : np.ndarray
        The logarithm of the formation constants.
    stoichiometry : np.ndarray
        The stoichiometric matrix.
    T : np.ndarray
        The total concentration.
    lmax : np.ndarray
        The maximum step allowed.
    g0 : np.ndarray
        The value of the function at λ=0.
    g2 : np.ndarray
        The value of the function at λ=0.99λ(max).

    Returns
    -------
    lmin : np.ndarray
        The minimum step.
    gmin : np.ndarray
        The value of the function at the minimum step.
    """
    nerr = np.geterr()
    np.seterr(all="ignore")

    def g(lambda_):
        "Auxiliary function."
        FF = fobj(
            species_concentration(
                x0 + lambda_[:, np.newaxis] * dx, log_beta, stoichiometry, full=True
            ),
            stoichiometry,
            T,
        )
        return 0.5 * np.sum(np.square(FF), axis=1)

    if lmax is None:
        lmax = -x0 / dx  # may cause division by 0
        lmax[lmax < 0.0] = 1.0
        lmax = np.min(lmax, axis=1)

    if g0 is None:
        g0 = g(np.zeros_like(lmax))

    g1 = g(lmax / 2)
    x1 = x0 + lmax[:, None] * dx
    x1[x1 < 0.0] = 0.0
    if g2 is None:
        g2 = g(0.99 * lmax)

    b = -g2 + 4 * g1 - 3 * g0
    a = (g2 - g0 - b) / lmax
    lmin = -0.5 * b / a  # may cause division by 0

    # In the unlikely case where a == 0.0 meaning g0 == g1 == g2 we set the
    # step halfway.
    w = a == 0.0
    if np.any(w):
        lmin[w] = lmax[w] / 2

    w = lmin < 0.1 * lmax  # Set minimum step as 0.1λ(max)
    lmin[w] = 0.1 * lmax[w]

    w = lmin > lmax  # In the unlikely case where λ(min)>λ(max), set
    lmin[w] = 0.95 * lmax[w]  # λ(min) close enough to λ(max)

    gmin = g(lmin)

    # Check g(lmin) < g0
    w2 = gmin > g0
    if log_beta.ndim == 1:
        _beta = log_beta
    else:
        _beta = log_beta[w2]
    if np.any(w2):
        lmin[w2], gmin[w2] = linesearch3(
            x0[w2],
            dx[w2],
            _beta,
            stoichiometry,
            T[w2],
            lmax=lmax[w2] / 2,
            g0=g0[w2],
            g2=gmin[w2],
        )
    np.seterr(**nerr)
    return lmin, gmin


def limit_step(x, dx):
    r"""
    Limit step.

    Given a state (x) and a step (dx), the next state is expected to be
    (x+dx). However, in some cases negative values are forbidden. This
    may happen for small values of the state. In the case of x being small
    we can approximate ($x+1 \approx e^{-x}$)

    Parameters
    ----------
    x : numpy.ndarray
        The state. It must be 1D.
    dx : numpy.ndarray
        The step. It must have the same length as x.

    Returns
    -------
    new_dx : numpy.ndarray
        The limited step.
    """
    if len(x) != len(dx):
        raise ValueError("both arguments must have the same size")
    # return np.where(who, x+dx, x*np.exp(dx))
    one_over_del = (-dx * x) / (0.5 * x)
    rev_del = 1 / np.where(one_over_del > 1, one_over_del, 1)

    return dx * rev_del


def DRScaling(J, F):
    r"""Apply scaling to both jacobian and objective function.

    Applies scaling technique according to De Robertis, *et al.* [^1]
    The scaling technique overcomes the problem of divergence
    when the jacobian (here called G) matrix is near singular.
    "... scaling was applied to matrix *G* and to vector *e*
    (residuals, $e = C_{k, calcd} - C_k$
    according to the equations $g_{kj}^* = g_{kj}(g_{kk}g_{jj})^{-1/2}$
    and $(e_k^* = e_kg_{kk}^{-1/2}$ where $g_{kj}^*)$
    and $(e_k^*)$ are the elements of the scaled matrix and vector
    respectively."

    $$
    J^*_{kj} = J_{kj}(J_{kk}J_{jj})^{-1/2}\\
    F^*_{k} = F_{k}J_{kk}^{-1/2}
    $$

    [^1]: De Robertis, *et al.*: *Analytica Chimica Acta* 1986, **191**, 385-398


    Parameters
    ----------
    J : numpy.ndarray
        jacobian array, which will be modified.
        It can be of any dimensionality provided that the last two are
        of the same size.
    F : numpy.ndarray
        residuals array, which will be modified.
        If must have one dimmension less than J and the rest of the
        axes be of the same size than J.

    Returns
    -------
    d : numpy.ndarray
        The diagonal of the jacobian, $(J_{ii})$
        to scale back the result.
    """
    d = J[np.arange(J.shape[0])[:, None], np.eye(J.shape[1], dtype=bool)]
    J /= np.sqrt(d[..., np.newaxis] * d[..., np.newaxis, :])
    F /= np.sqrt(d)
    return d


def fobj(
    concentration,
    log_ks,
    stoichiometry,
    solid_stoichiometry,
    total_concentration,
):
    """
    Calculate the objective function for a given set of parameters.

    Parameters
    ----------
    concentration : numpy.ndarray
        The concentration array of shape (n, c+p), where n is the number of points c is the number of components and p is the number of solid species.
    log_ks : numpy.ndarray
        The logarithm of the solubility products with shape (n, p), where p is the number of solid species.
    stoichiometry : numpy.ndarray
        The stoichiometry matrix with shape (n, s), where s is the number of soluble species.
    solid_stoichiometry : numpy.ndarray
        The stoichiometry matrix for solid species with shape (n, p), where p is the number of solid species.
    total_concentration : numpy.ndarray
        The total concentration vector with shape (n,c), where n is the number of points and c is the number of components.

    Returns
    -------
    delta : numpy.ndarray
        The objective function values with shape (n, nt), where nt is the total number of components (n + f).
    """
    nc = stoichiometry.shape[0]
    nf = concentration.shape[1] - nc - stoichiometry.shape[1]

    c_components = concentration[:, :nc]
    c_solids = concentration[:, nc : nc + nf]
    c_species = concentration[:, nc + nf :]

    components_in_species = c_species[:, np.newaxis, :] * stoichiometry[np.newaxis]
    components_in_species = np.sum(components_in_species, axis=2)

    if c_solids.size > 0:
        components_in_solids = (
            c_solids[:, np.newaxis, :] * solid_stoichiometry[np.newaxis]
        )
        components_in_solids = np.sum(components_in_solids, axis=2)
    else:
        components_in_solids = 0

    delta = (
        c_components
        + components_in_species
        + components_in_solids
        - total_concentration
    )
    if nf > 0:
        solid_delta = np.log10(c_components) @ solid_stoichiometry - log_ks
    else:
        solid_delta = np.empty((delta.shape[0], 0))

    delta = np.concatenate((delta, solid_delta), axis=1)

    return delta


def jacobian(concentration, stoichiometry, solid_stoichiometry):
    """
    Compute the Jacobian matrix for the given system of equations.

    Parameters
    ----------
    concentration : numpy.ndarray
        The concentration array of shape (n, c+p), where n is the number of points c is the number of components and p is the number of solid species.
    stoichiometry : numpy.ndarray
        The stoichiometry matrix of shape (c, s), representing the stoichiometric coefficients of the soluble components.
    solid_stoichiometry : numpy.ndarray
        The solid stoichiometry matrix of shape (n, p), representing the stoichiometric coefficients of the solid components.

    Returns
    -------
    numpy.ndarray
        The Jacobian matrix of shape (n, nt, nt), where nt is the total number of components (n + f).

    """
    nt = nc = stoichiometry.shape[0]
    nf = solid_stoichiometry.shape[1]
    nt += nf

    J = np.zeros(shape=(concentration.shape[0], nt, nt))
    diagonals = np.einsum(
        "ij,jk->ijk", concentration[:, nt:], np.eye(concentration.shape[1] - nt)
    )
    # Compute Jacobian for soluble components only
    J[:, :nc, :nc] = stoichiometry @ diagonals @ stoichiometry.T
    J[:, range(nc), range(nc)] += concentration[:, :nc]

    # Add solid contribution if necessary
    if nf > 0:
        J[:, nc:nt, :nc] = solid_stoichiometry.T
        J[:, :nc, nc:nt] = solid_stoichiometry
    return J
