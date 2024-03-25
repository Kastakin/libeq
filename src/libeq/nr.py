import numpy as np
from .species_conc import species_concentration
from .damping import damping as _damping
from .excepts import TooManyIterations, FailedCalculateConcentrations


def NewtonRaphson(
    x0,
    *,
    log_beta,
    log_ks,
    stoichiometry,
    solid_stoichiometry,
    total_concentration,
    solids_idx=[],
    max_iterations=1000,
    threshold=1e-10,
    damping=False,
    forcer=False,
    scaling=False,
    step_limiter=True,
    zero_offdiag=False,
    logc=False,
    debug=False,
    panic=True,
    **kwargs,
):
    r"""Solve the set of equations J·δc = -F using Newton-Raphson's method.

    Given an initial guess **x0** and the parameters of the titration, this
    function uses the Newton-Raphson method to solve the free
    concentrations.  It includes several
    convergence tecniques such as scaling, damping and step limiting. For the
    original source of damping and scaling see De Robertis, *et al.*,
    [#f1]_ and also Del Piero, *et al.* [#f2]_

    Parameters:
        x0 (:class:`numpy.ndarray`): initial guess for iterations. If *logc* is
            True, the natural logarithm of the concentrations is expected to be
            input. Otherwise, the regular concentration are expected.
        beta (:class:`numpy.ndarray`): values of equilibrium constants
        stoichiometry (:class:`numpy.ndarray`): stoichiometric coefficients
        T (:class:`numpy.ndarray`): total concentration values
        scaling (bool, optional): Whether to use or not scaling techniques.
            False by default. See :func:`DRscaling`.
        damping (bool, optional): Whether to use or not damping techniques.
            False by default. See :func:`DRdamping`.
        step_limiter (bool, optional[True]): Whether to use step limiter or
            not.  Step limiter attempts to avoid a step which leads to negative
            concentrations. It searches the increments which will result in
            negative concentrations and replaces the step
            :math:`x_n = x_{n-1} + dx` for
            :math:`x_n = x_{n+1} \cdot e^{dx}`
            which always gives a positive concentration as a result.
            It can jointly be used with the forcer.
        forcer (bool, optional[False]): Whether to use forcer techniques.
            False by default.
        threshold (float): threshold criterium for convergence
        max_iterations (int): maximum number of iterations allowed
        zero_offdiag (bool): If this option is applied, all non-diagonal
            elements of the jacobian are set to zero. This option is useful
            when only an estimate of the free concentrations is wanted.
        max_damps (int):  default, 2. maximum number of dampimgs allowed.
            It applies only if damping = True.
        logc (bool): Fit the logarithmic value of the concentration
        debug (bool): The handle for logging
        do_iterations (int, optional): Perform exatly *do_iterations*
            iterations and do not test for convergence.
        panic (bool, optional, default True): If convergence fails, dump to a
            file the data to debug.
    Returns:
        :class:`numpy.ndarray`: An array containing the values of the free
            concentrations.
    Raises:
        :class:`RuntimeError`: if the concentrations cannot be calculated
        :class:`excepts.TooManyIterations`: if the maximum number of iterations
            is reached.

    .. warning:: This function is the very core of the program. Do not change
        the default parameters unless you know what you are doing.

    .. [#f1] *Analytica Chimica Acta* 1986, **191**, 385-398
    .. [#f2] Del Piero, *et al.*: *Annali di Chimica* 2006, 96.
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
            "Options scaling and zero_offdiag are not" + "compatible with each other"
        )

    if "do_iterations" in kwargs:
        do_iterations = kwargs["do_iterations"]
        if not isinstance(do_iterations, int):
            raise TypeError("do_iteration must be a positive int.")
        if do_iterations < 1:
            raise ValueError("do_iteration must be a positive int.")
    else:
        do_iterations = None

    n_species = stoichiometry.shape[1]

    if solids_idx:
        solids = True
    else:
        solids = False

    # copy x0 so that it is not modified outside this function.
    x = np.copy(x0)
    # ------ check input --------
    # libaux.assert_array_dim(2, x0)
    x, total_concentration = np.atleast_2d(x, total_concentration)

    # ------ main loop ----------
    for iterations in range(max_iterations):
        c0 = species_concentration(
            x, log_beta, stoichiometry, solid_stoichiometry, full=True, logc=logc
        )
        if logc:
            _c = 10 ** (c0)
        else:
            _c = c0

        F = fobj(
            _c,
            log_beta,
            log_ks,
            stoichiometry,
            solid_stoichiometry,
            total_concentration,
        )
        J = jacobian(_c, stoichiometry, solid_stoichiometry, solids=solids, logc=logc)

        if solids:
            J = np.delete(J, solids_idx, axis=0)
            J = np.delete(J, solids_idx, axis=1)
            F = np.delete(F, solids_idx, axis=0)

        if np.any(np.isnan(J)):
            _panic_save()
            msg2 = f"could not calculate jacobian (iteration {iterations})"
            raise FailedCalculateConcentrations(msg2, x)
        if np.any(np.isnan(F)):
            msg2 = f"could not calculate residuals (iteration {iterations})"
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
            x += limit_step(x, dx) * x
        else:
            x += dx * x

        if (do_iterations and iterations + 1 >= do_iterations) or np.all(
            np.abs(F) < threshold
        ):
            return x, log_beta

        if damping:
            x = _damping(
                x, log_beta, stoichiometry, solid_stoichiometry, total_concentration
            )

    raise TooManyIterations("too many iterations", x)


def linesearch3(x0, dx, log_beta, stoichiometry, T, lmax=None, g0=None, g2=None):
    r"""Three-point parabolic line search.

    This functions implements a 3-point linesearch in the Newton direction.
    This is a variation of the line search for 2 points.[#]_ The function to
    be minimized is the same though but the approach is different and it is
    adapted to the nature of the problem of concentration solving. We define
    a function :math:`f=\frac12F\cdot F` which is to be minimized. Then we
    define a parameter λ that 0<λ<1 which is the fraction of the Newton step
    and then another function *g* which is function of the fractional is
    defined so that

    .. math:: g(\lambda) = f(x_0 + \lambda \delta x)

    We know that negative x₀ values are forbidden, therefore λ might limited
    to values lower than 1. The maximum value allowed for λ is that that makes
    any concentration equal to 0, therefore
    :math:`\lambda_{max} = -x_0/\delta` if :math:`-x_0/\delta<1`

    We model :math:`g(\lambda)` as a parabolic function for which we calculate
    the values for λ=0, λ=½λ(max) and λ=0.99λ(max).

    .. [#] W. H. Press, S. A. Teukolksy, W. T. Vetterling, Brian P. Flannery,
       Numerical Recipes in C. The Art of Scientific Computing, Second Edition
       1997, pages 384--385.
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
    r"""Limit step.

    Given a state (**x**) and a step (**dx**), the next state is expected to be
    :math:`x+dx`. However, in some cases negative values are forbidden. This
    may happen for small values of the state. In the case of *x* being small
    we can approximate :math:`x+1 \simeq e^{-x}`

    Parameters:
        x (:class:`numpy.ndarray`): The state. It must be 1D.
        dx (:class:`numpy.ndarray`): The step. It must have  the same length
            than *x*.

    """
    if len(x) != len(dx):
        raise ValueError("both arguments must have the same size")
    # return np.where(who, x+dx, x*np.exp(dx))
    one_over_del = (-dx * x) / (0.5 * x)
    rev_del = 1 / np.where(one_over_del > 1, one_over_del, 1)

    return dx * rev_del


def DRScaling(J, F):
    """Apply scaling to both jacobian and objective function.

    Applies scaling technique according to De Robertis, *et al.* [#f1]_
    The scaling technique overcomes the problem of divergence
    when the jacobian (here called G) matrix is near singular.
    "... scaling was applied to matrix *G* and to vector *e*
    (residuals, :math:`e = C_{k, calcd} - C_k`)
    according to the equations
    :math:`g_{kj}^* = g_{kj}(g_{kk}g_{jj})^{-1/2}`
    and
    :math:`e_k^* = e_kg_{kk}^{-1/2}` where :math:`g_{kj}^*`
    and :math:`e_k^*` are the elements of the scaled matrix and vector
    respectively."

    .. math:: J^*_{kj} = J_{kj}(J_{kk}J_{jj})^{-1/2}
    .. math:: F^*_{k} = F_{k}J_{kk}^{-1/2}

    Parameters:
        J (:class:`numpy.ndarray`): jacobian array, which will be modified.
            It can be of any dimensionality provided that the last two are
            of the same size.
        F (:class:`numpy.ndarray`): residuals array, which will be modified.
            If must have one dimmension less than J and the rest of the
            axes be of the same size than J.

    Returns:
        :class:`numpy.ndarray`: The diagonal of the jacobian, :math:`J_{ii}`
            to scale back the result.
    """
    d = J[np.arange(J.shape[0])[:, None], np.eye(J.shape[1], dtype=bool)]
    J /= np.sqrt(d[..., np.newaxis] * d[..., np.newaxis, :])
    F /= np.sqrt(d)
    return d


def fobj(
    concentration,
    log_beta,
    log_ks,
    stoichiometry,
    solid_stoichiometry,
    total_concentration,
    solids=False,
):
    nc = stoichiometry.shape[0]

    c1 = concentration[:, :nc]
    c2 = concentration[:, nc:]

    delta = (
        c1
        + np.sum(c2[:, np.newaxis, :] * stoichiometry[np.newaxis], axis=2)
        - total_concentration
    )

    if solids:
        solid_delta = np.log10(concentration[:, :nc]) @ solid_stoichiometry - log_ks
        delta = np.concatenate((delta, solid_delta))

    return delta


def jacobian(
    concentration, stoichiometry, solid_stoichiometry, solids=False, logc=False
):
    nc = stoichiometry.shape[0]
    nt = nc

    if solids:
        nf = solid_stoichiometry.shape[0]
        nt += nf

    J = np.zeros(shape=(concentration.shape[0], nt, nt))
    diagonals = np.einsum(
        "ij,jk->ijk", concentration[:, nc:], np.eye(concentration.shape[1] - nc)
    )
    # Compute Jacobian for soluble components only
    J[:, :nc, :nc] = stoichiometry @ diagonals @ stoichiometry.T
    J[:, range(nc), range(nc)] += concentration[:, :nc]

    # Add solid contribution if necessary
    if solids:
        J[nc:nt, :nc] = solid_stoichiometry.T
        J[:nc, nc:nt] = solid_stoichiometry
    return J
