from functools import partial
from itertools import accumulate

import numpy as np

from libeq.data_structure import SolverData
from libeq.solver import solve_equilibrium_equations
from libeq.solver.solver_utils import (
    _assemble_outer_fixed_point_params,
    _prepare_common_data,
    _titration_total_c,
)

from .fitter import levenberg_marquardt


def PotentiometryOptimizer(data: SolverData, reporter=None):
    def f_obj(c):
        """
        Given the concentrations of the components, calculate the objective function value.

        Parameters:
        -------
        x : numpy.ndarray
            The concentrations of the components.

        Returns:
        -------
        emf : numpy.ndarray
            The calculcated potential from components.

        """
        electroactive = fhsel(c)
        calc_remf = np.log(electroactive)
        return np.ravel(calc_remf)

    def free_conc(updated_beta):
        nonlocal _initial_guess
        incoming_beta = updated_beta / 2.303
        gen = ravel(data.log_beta, incoming_beta, beta_flags)
        log_beta = np.fromiter(gen, dtype=float)
        # Solve the system of equations
        c, *_ = solve_equilibrium_equations(
            stoichiometry=stoichiometry,
            solid_stoichiometry=solid_stoichiometry,
            original_log_beta=log_beta,
            original_log_ks=original_log_ks,
            total_concentration=total_concentration,
            outer_fiexd_point_params=outer_fixed_point_params,
            initial_guess=_initial_guess,
            full=True,
        )
        _initial_guess = c[:, : stoichiometry.shape[0]]
        return c

    def jacobian(concentration):
        """
        Calculate the jacobian matrix of the objective function.

        Parameters:
        -------
        x : numpy.ndarray
            The concentrations of the components.

        Returns:
        -------
        jac : numpy.ndarray
            The jacobian matrix of the objective function.

        """
        nc = stoichiometry.shape[0]
        J = np.zeros(shape=(concentration.shape[0], nc, nc))
        diagonals = np.einsum(
            "ij,jk->ijk", concentration[:, nc:], np.eye(concentration.shape[1] - nc)
        )
        # Compute Jacobian for soluble components only
        J = stoichiometry @ diagonals @ stoichiometry.T
        J[:, range(nc), range(nc)] += concentration[:, :nc]

        B = stoichiometry[np.newaxis, ...] * concentration[..., np.newaxis, nc:]
        dcdb = np.squeeze(np.linalg.solve(J, -B))
        return fhsel(dcdb[..., np.flatnonzero(beta_flags)]).T

    def text_reporter(*args):
        print(f"iteration n.{args[0]}")
        print("x", args[1])
        print("dx", args[2])
        print("sigma", args[3])
        print("----------------\n")

    # Load the n titrations with their potential from the data file
    emf = [t.emf for t in data.potentiometry_options.titrations]
    emf0 = [t.e0 for t in data.potentiometry_options.titrations]
    slope = [t.slope for t in data.potentiometry_options.titrations]
    v_add = [t.v_add for t in data.potentiometry_options.titrations]

    ll, ul = data.potentiometry_options.px_range

    reduced_emf = [
        build_reduced_emf(emf_, emf0_, slope_)
        for emf_, emf0_, slope_ in zip(emf, emf0, slope)
    ]
    if ul + ll != 0:
        idx_to_keep = [
            (-red_emf >= ll * 2.303) & (-red_emf <= ul * 2.303)
            for red_emf in reduced_emf
        ]
        reduced_emf = [red_emf[idx] for red_emf, idx in zip(reduced_emf, idx_to_keep)]
        emf = [emf[idx] for emf, idx in zip(emf, idx_to_keep)]
        v_add = [v_add[idx] for v_add, idx in zip(v_add, idx_to_keep)]
    else:
        idx_to_keep = [None for _ in reduced_emf]

    full_emf = np.concatenate(reduced_emf, axis=0).ravel()
    n_exp_points = full_emf.shape[0]

    if data.potentiometry_options.weights == "constants":
        weights = np.ones(n_exp_points)
    elif data.potentiometry_options.weights == "calculated":
        e0_sigma = [t.e0_sigma for t in data.potentiometry_options.titrations]
        v0_sigma = [t.v0_sigma for t in data.potentiometry_options.titrations]

        weights = np.concatenate(
            [
                compute_weights(emf_, v_add_, e0_sigma_, v0_sigma_)
                for emf_, v_add_, e0_sigma_, v0_sigma_ in zip(
                    emf, v_add, e0_sigma, v0_sigma
                )
            ],
            axis=0,
        ).ravel()

    elif data.potentiometry_options.weights == "given":
        raise NotImplementedError("User given weights are not implemented yet.")

    slices = list(accumulate([0] + [s.shape[0] for s in reduced_emf]))
    electro_active_components = [
        t.electro_active_compoment for t in data.potentiometry_options.titrations
    ]
    fhsel = partial(hselect, hindices=electro_active_components, slices=slices[:-1])

    beta_flags = np.array(data.potentiometry_options.beta_flags).astype(int)
    beta_flags = np.where(beta_flags == -1, 0, beta_flags)

    (
        stoichiometry,
        solid_stoichiometry,
        original_log_beta,
        original_log_ks,
        charges,
        independent_component_activity,
    ) = _prepare_common_data(data)

    total_concentration = np.vstack(
        [
            _titration_total_c(t, i)
            for t, i in zip(data.potentiometry_options.titrations, idx_to_keep)
        ]
    )

    original_log_beta = np.tile(original_log_beta, (total_concentration.shape[0], 1))
    original_log_ks = np.tile(original_log_ks, (total_concentration.shape[0], 1))

    outer_fixed_point_params = _assemble_outer_fixed_point_params(
        data, charges, independent_component_activity
    )

    _initial_guess, *_ = solve_equilibrium_equations(
        stoichiometry=stoichiometry,
        solid_stoichiometry=solid_stoichiometry,
        original_log_beta=original_log_beta,
        original_log_ks=original_log_ks,
        total_concentration=total_concentration,
        outer_fiexd_point_params=outer_fixed_point_params,
        initial_guess=None,
        full=False,
    )

    if outer_fixed_point_params["ionic_strength_dependence"] is True:
        print(
            "Ionic strength dependence for potentiometry oprimization is not implemented yet."
        )
        outer_fixed_point_params["ionic_strength_dependence"] = False

    x, concs, return_extra = levenberg_marquardt(
        np.fromiter(unravel(data.log_beta, beta_flags), dtype=float) * 2.303,
        full_emf,
        f_obj,
        free_conc,
        jacobian,
        weights,
        report=reporter,
    )

    return_extra["slices"] = slices

    return_extra["read_potential"] = emf

    reduced_calculated_emf = f_obj(concs)
    ix_ranges = list(zip(slices, slices[1:] + [concs.shape[0]]))[:-1]
    calculated_potential = []
    residuals_potential = []
    for counter, (i1, i2) in enumerate(ix_ranges):
        calculated_potential.append(
            rebuild_emf(reduced_calculated_emf[i1:i2], emf0[counter], slope[counter])
        )
        residuals_potential.append(emf - calculated_potential[-1])
    return_extra["calculated_potential"] = calculated_potential
    return_extra["residuals_potential"] = residuals_potential

    b_error, cor_matrix, cov_matrix = fit_final_calcs(
        return_extra["jacobian"], return_extra["residuals"], return_extra["weights"]
    )

    return x, concs, b_error, cor_matrix, cov_matrix, return_extra


def build_reduced_emf(emf, emf0, slope):
    """
    Build the reduced emf array from the emf, emf0, and slope values.

    Parameters:
    -------
    emf : numpy.ndarray
        The emf values.
    emf0 : float
        The standard emf value.
    slope : float
        The slope.

    Returns:
    -------
    reduced_emf : numpy.ndarray
        The reduced emf values.

    """
    return (emf - emf0) / (slope / 2.303)


def rebuild_emf(remf, emf0, slope):
    """
    Reuild the original emf from the reduced emf, emf0, and slope values.

    Parameters:
    -------
    remf : numpy.ndarray
        The reduced emf values.
    emf0 : float
        The standard emf value.
    slope : float
        The slope.

    Returns:
    -------
    emf : numpy.ndarray
        The emf values.

    """
    return (remf * (slope / 2.303)) + emf0


def compute_weights(emf, v_add, e_sigma, v_sigma):
    """
    Compute the weights for the given emf, v_add, e_sigma, and v_sigma values.

    Parameters:
    -------
    emf : numpy.ndarray
        The emf values.
    v_add : numpy.ndarray
        The v_add values.
    e_sigma : numpy.ndarray
        The e_sigma values.
    v_sigma : numpy.ndarray
        The v_sigma values.

    Returns:
    -------
    weights : numpy.ndarray
        The calculated weights.

    """
    der2 = np.gradient(emf, v_add) ** 2
    return 1 / (der2 * v_sigma**2 + e_sigma**2)


def hselect(array, hindices, slices):
    """Select columns that correspond to the electroactive species.

    Given the concentrations array, selects the columns that correspond
    to the electroactive species.

    Parameters:
        array (:class:`numpy.ndarray`): The :term:`free concentrations array`
        hindices (list): List of ints or list of lists of ints with the indices
            of the electroactive species. Example: [[0,1],[1,2],[3,4],[4,5]].
            hindices are applied along axis=0
        slices (list of ints): Where to divide C. Example: [ 0, 5, 10, 15 ]
            slices are applied along axis=1

    Returns:
        The part of C which is electroactive

    >>> slices = [0, 4, 7]
    >>> hindices = [[0,1],[1,2],[3,4]]
    >>> C = np.array([[ 0.255,  0.638,  0.898,  0.503,  0.418],
    ...               [ 0.383,  0.789,  0.731,  0.713,  0.629],
    ...               [ 0.698,  0.080,  0.597,  0.503,  0.456],
    ...               [ 0.658,  0.399,  0.332,  0.700,  0.294],
    ...               [ 0.534,  0.556,  0.762,  0.493,  0.510],
    ...               [ 0.637,  0.065,  0.638,  0.770,  0.879],
    ...               [ 0.598,  0.193,  0.912,  0.263,  0.118],
    ...               [ 0.456,  0.680,  0.049,  0.381,  0.872],
    ...               [ 0.418,  0.456,  0.430,  0.842,  0.172]])
    >>> hselect(C, hindices, slices)
    array([[0.255, 0.638], [0.383, 0.789], [0.698, 0.080], [0.658, 0.399],
           [0.556, 0.762], [0.065, 0.638], [0.193, 0.912], [0.381, 0.872],
           [0.842, 0.172]])
    """
    if slices is None and isinstance(int, hindices):
        return array[:, hindices, ...]

    if len(hindices) != len(slices):
        raise TypeError("hindices and slices have wrong size")
    # libaux.assert_array_dim(2, array)

    # slices → [ 0, 5, 10, 15 ]
    #          0→4 5→9  10→14  15→end
    # hindices → [[0,1],[1,2],[3,4],[4,5]]
    # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
    # -------------  ------------- -------------- --------
    # 0  0  0  0  0  1  1  1  1  1  3  3  3  3  3  4  4  4
    # 1  1  1  1  1  2  2  2  2  2  4  4  4  4  4  5  5  5

    num_points = array.shape[0]
    nslices = (b - a for a, b in zip(slices, slices[1:] + [array.shape[0]]))
    # y = np.array(sum([n*h for n, h in zip(nslices, hindices)], []))
    y = np.vstack([np.tile(np.array(h), (n, 1)) for h, n in zip(hindices, nslices)])
    return np.squeeze(array[np.arange(num_points), y.T, ...].T)


def unravel(x, flags):
    """Unravel data according to flags provided.

    This routine takes an array of data and an array of flags of the same
    length and returns another array with only the independet variables.

    Parameters:
        x (iterable): the original data values
        flags (iterable): the flags indicating how to update x.
            Values must int. Accepted values are

            * 0: value is to be kept constant
            * 1: value is to be refined and the corresponding value from x
              will be substituted by the corresponding value from y.
            * >2: value is restrained. All places with the same number are
              refined together and the ratio between them is maintained.

    Returns:
        generator: Values of **x** processed according to **flags**.
    """
    constr_list = []
    for i, f in zip(x, flags):
        if f == 1:
            yield i
        if f > 1:
            if f not in constr_list:
                yield i
                constr_list.append(f)


def ravel(x, y, flags):
    """Update values from one iterable with other iterable according to flags.

    This function does the opposite action than :func:`unravel`.

    Parameters:
        x (iterable): the original array values
        y (iterable): the updated values to be plugged into *x*.
        flags (sequence): flags indicating how to update *x* with *y*. Accepted
            values are:

            * 0: value is to be kept constant
            * 1: value is to be refined and the corresponding value from x
              will be substituted by the corresponding value from y.
            * >2: value is restrained. All places with the same number are
              refined together and the ratio between them is maintained.

    Yields:
        float: Raveled values.
    """
    # indices of the reference parameter for constraining
    ref_index = {i: flags.index(i) for i in range(2, 1 + max(flags))}
    ref_val = {}

    ity = iter(y)
    for i, f in enumerate(flags):
        if f == 1:  # refinable: return new value
            yield next(ity)
        elif f == 0:  # constant: return old value
            yield x[i]
        else:  # constrained: return or compute
            if i == ref_index[f]:
                val = next(ity)  # reference value: return new value
                ref_val[f] = val  # and store ref value
                yield val
            else:  # other: compute proportional value
                yield x[i] * ref_val[f] / x[ref_index[f]]


def covariance_fun(J, W, F):
    """Compute covariance matrix.

    Returns the covariance matrix :math:`CV = inv(J'.W.J)*MSE`
    Where MSE is mean-square error :math:`MSE = (R'*R)/(N-p)`
    where *R* are the residuals, *N* is the number of observations and
    *p* is the number of coefficients estimated

    Parameters:
        J (:class:`numpy.ndarray`): the jacobian
        W (:class:`numpy.ndarray`): the weights matrix
        F (:class:`numpy.ndarray`): the residuals
    Returns:
        :class:`numpy.ndarray`: an (*p*, *p*)-sized array representing
            the covariance matrix.
    """
    mse = np.sum(F * np.diag(W) * F) / (len(F) - J.shape[1])
    temp = np.linalg.inv(np.dot(np.dot(J.T, W), J))
    return temp * mse


def fit_final_calcs(jacobian, resids, weights):
    """Perform final calculations common to some routines.

    Parameters:
        jacobian (:class:`numpy.array`): the jacobian
        resids (:class:`numpy.array`): the residuals
        weights (:class:`numpy.array`): the weights
    Returns:
        * the error in beta
        * the correlation matrix
        * the covariance matrix
    """
    covariance = covariance_fun(jacobian, weights, resids)
    cov_diag = np.diag(covariance)
    error_B = np.sqrt(cov_diag) / np.log(10)
    lenD = len(cov_diag)
    correlation = covariance / np.sqrt(
        np.dot(cov_diag.reshape((lenD, 1)), cov_diag.reshape((1, lenD)))
    )
    return error_B, correlation, covariance
