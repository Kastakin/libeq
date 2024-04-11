from typing import Literal
import numpy as np
from numpy.typing import NDArray

from .data_structure import SolverData
from .damping import damping
from .nr import NewtonRaphson
from .wrappers import outer_fixed_point


def EqSolver(
    data: SolverData, mode: Literal["titration", "distribution"] = "titration"
):
    """
    Solve the equilibrium equations for a given set of data.

    Args:
        data (SolverData): The input data containing the necessary information for the solver.
        mode (Literal["titration", "distribution"], optional): The mode of operation for the solver.
            Defaults to "titration".

    Returns:
        Tuple: A tuple containing the result of the solver and the updated values of the equilibrium constants.

    Raises:
        None

    """
    # Read the data
    stoichiometry = data.stoichiometry
    solid_stoichiometry = data.solid_stoichiometry
    log_beta = data.log_beta
    log_ks = data.log_ks
    ionic_strength_dependence = data.ionic_strength_dependence
    charges = np.atleast_2d(np.concatenate((data.charges, data.species_charges)))

    # Get the total concentration values depending if mode is titration or distribution
    if mode == "titration":
        c0 = data.c0
        ct = data.ct
        v0 = data.v0
        v_add = data.v_add
        total_concentration: NDArray = (
            ((c0 * v0)[:, np.newaxis] + ct[:, np.newaxis] * v_add) / (v_add + v0)
        ).T

        independent_component_activity = None

    elif mode == "distribution":
        # TODO remove after testing
        data.ionic_strength_dependence = True
        if data.ionic_strength_dependence:
            print(
                "Ionic strength dependence is not implemented for distribution mode!\n No ionic strength dependence will be considered."
            )

        independent_component = data.distribution_opts.independent_component
        independent_component_concentration = 10 ** -np.arange(
            data.distribution_opts.initial_log,
            (data.distribution_opts.final_log + data.distribution_opts.log_increments),
            data.distribution_opts.log_increments,
        )
        total_concentration = np.repeat(
            data.c0[np.newaxis, :], len(independent_component_concentration), axis=0
        )
        total_concentration[:, independent_component] = (
            independent_component_concentration
        )
        # Reduce the problem by accounting for the independent component
        (
            independent_concentration,
            total_concentration,
            log_beta,
            log_ks,
            stoichiometry,
            solid_stoichiometry,
        ) = freeze_concentration(
            independent_component,
            total_concentration,
            log_beta,
            log_ks,
            stoichiometry,
            solid_stoichiometry,
            ionic_strength_dependence=ionic_strength_dependence,
        )
        independent_component_charge = charges[:, independent_component]
        independent_component_activity = 0.5 * (
            independent_concentration * (independent_component_charge**2)
        )
        charges = np.delete(
            charges, data.distribution_opts.independent_component, axis=1
        )

    outer_fiexd_point_params = [
        data.ionic_strength_dependence,
        charges,
        data.ref_ionic_str,
        data.dbh_values,
    ]
    damping_fn = outer_fixed_point(
        *outer_fiexd_point_params,
        independent_component_activity=independent_component_activity,
    )(damping)

    nr_fn = outer_fixed_point(
        *outer_fiexd_point_params,
        independent_component_activity=independent_component_activity,
    )(NewtonRaphson)

    # Get the initial guess for the free concentrations
    initial_guess, log_beta = damping_fn(
        np.full_like(total_concentration, 1e-6),
        log_beta=log_beta,
        log_ks=log_ks,
        stoichiometry=stoichiometry,
        solid_stoichiometry=solid_stoichiometry,
        total_concentration=total_concentration,
        tol=1e-3,
    )

    # Add the solid concentrations to the initial guess
    initial_guess = np.concatenate(
        (initial_guess, np.full((initial_guess.shape[0], data.nf), 0)), axis=1
    )

    # Apply Newton-Raphson iterations
    result, log_beta = nr_fn(
        initial_guess,
        log_beta=log_beta,
        log_ks=log_ks,
        stoichiometry=stoichiometry,
        solid_stoichiometry=solid_stoichiometry,
        total_concentration=total_concentration,
        max_iterations=1000,
        threshold=1e-10,
    )
    if data.nf > 0:
        result, log_beta, log_ks, saturation_index = solids_solver(
            result,
            log_beta,
            log_ks,
            stoichiometry,
            solid_stoichiometry,
            total_concentration,
            outer_fiexd_point_params=outer_fiexd_point_params,
            independent_component_activity=independent_component_activity,
        )

    return result, log_beta


def solids_solver(
    concentrations: NDArray,
    log_beta,
    log_ks,
    stoichiometry,
    solid_stoichiometry,
    total_concentration,
    outer_fiexd_point_params,
    independent_component_activity=None,
):
    final_result = np.empty_like(concentrations)
    final_log_beta = np.empty_like(log_beta)
    final_log_ks = np.empty_like(log_ks)
    final_saturation_index = np.empty_like(log_ks)
    for point, c in enumerate(concentrations):
        solids_set = set()
        (
            point_log_beta,
            point_log_ks,
            point_total_concentration,
            point_independent_component_activity,
        ) = _get_point_values(
            log_beta, log_ks, total_concentration, independent_component_activity, point
        )

        c = np.atleast_2d(c)

        saturation_index = compute_saturation_index(
            c, log_ks[[point], :], solid_stoichiometry
        )
        adjust_solids = (saturation_index > 1).any(axis=1)

        newton_raphson_solver = outer_fixed_point(
            *outer_fiexd_point_params,
            independent_component_activity=point_independent_component_activity,
        )(NewtonRaphson)

        while adjust_solids:
            solids_set, adjust_solids = _update_solids_set(
                total_concentration, c, point_log_ks, saturation_index, solids_set
            )

            if not adjust_solids:
                break

            c, point_log_beta = newton_raphson_solver(
                c,
                log_beta=point_log_beta,
                log_ks=point_log_ks,
                stoichiometry=stoichiometry,
                solid_stoichiometry=solid_stoichiometry,
                solids_idx=list(solids_set),
                total_concentration=point_total_concentration,
                max_iterations=1000,
                threshold=1e-8,
            )
            saturation_index = compute_saturation_index(
                c, log_ks[[point], :], solid_stoichiometry
            )
            adjust_solids = (saturation_index > 1).any(axis=1)

        final_result[point] = c
        final_log_beta[point] = point_log_beta
        final_log_ks[point] = point_log_ks
        final_saturation_index[point] = saturation_index

    return final_result, final_log_beta, final_log_ks, final_saturation_index


def _get_point_values(
    log_beta, log_ks, total_concentration, independent_component_activity, point
):
    point_log_beta = log_beta[[point], :]
    point_log_ks = log_ks[[point], :]
    point_total_concentration = total_concentration[[point], :]
    if independent_component_activity is not None:
        point_independent_component_activity = independent_component_activity[[point]]
    else:
        point_independent_component_activity = None
    return (
        point_log_beta,
        point_log_ks,
        point_total_concentration,
        point_independent_component_activity,
    )


def _update_solids_set(
    total_concentration, c, point_log_ks, saturation_index, solids_set
):
    adjust_solids = True
    negative_solid_concentration = c[:, point_log_ks.shape[1] :] < 0
    supersaturated_solid = saturation_index > 1 + 1e-9

    # any negative solid concentration remove them from the solids set
    if negative_solid_concentration.any():
        negative_solid_idx = (
            np.where(negative_solid_concentration)[1] + total_concentration.shape[1]
        )
        solids_set -= set(negative_solid_idx)
    elif supersaturated_solid.any():
        supersaturated_solid_idx = (
            np.where(supersaturated_solid)[1] + total_concentration.shape[1]
        )
        solids_set |= set(supersaturated_solid_idx)
    else:
        adjust_solids = False

    return solids_set, adjust_solids


def compute_saturation_index(concentrations, log_ks, solid_stoichiometry):
    r"""
    Compute the saturation index of the solid phases.

    Parameters:
    ----------
    concentrations : numpy.ndarray
        The concentrations of the free components.
    log_ks : numpy.ndarray
        The logarithm of the solubility product constants.
    solid_stoichiometry : numpy.ndarray
        The stoichiometric coefficient matrix for the solid phases.

    Returns:
    -------
    saturation_index : numpy.ndarray
        The saturation index of the solid phases.
    """
    nf = solid_stoichiometry.shape[1]
    nc = concentrations.shape[1] - nf
    saturation_index = 10 ** (
        np.log10(concentrations[:, :nc]) @ solid_stoichiometry - log_ks
    )
    return saturation_index


def freeze_concentration(
    independent_component,
    total_concentration,
    log_beta,
    log_ks,
    stoichiometry,
    solid_stoichiometry,
    ionic_strength_dependence=False,
):
    r"""Convert one component to independent variable.

    When solving the equilibrium, sometimes those concentrations are plotted
    as function of the concentration of one of them, typically the pH. That
    component, therefore, can be converted into an independent variable
    and removed from the unknowns.

    .. math::

    c_{i+S} = \beta_ic_{\omega}^{p_{i\omega}}\prod_{j\ne\omega}^Sc_j^{p_{ij}}
            = \beta_i'\prod_{j\ne\omega}^Sc_j^{p_{ij}}

    Parameters:
        beta (:class:`numpy.ndarray`): The equilibrium constants array.
        stoichiometry (:class:`numpy.ndarray`): The stoichiometric coefficient
            array
        analyticalc (:class:`numpy.ndarray`): Analytical concentrations array.
        reference (int):

    Returns:
        new_x (:class:`numpy.ndarray`): The reference component concentrations
            which is equal to the *analyltc* reference column.
        beta_prime (:class:`numpy.ndarray`): The new beta array.
        stoich_new (:class:`numpy.ndarray`): The new stoichiometry array with
            is equal to the original one with the reference component removed.
        analc_new (:class:`numpy.ndarray`): The new analytical concentrations
            array with is equal to the original one with the reference
            component removed.
    """
    new_x = total_concentration[:, independent_component]
    log_beta_prime = np.log10(
        (
            (10 ** log_beta[np.newaxis, :])
            * new_x[:, np.newaxis] ** stoichiometry[independent_component, :]
        )
    )
    stoich_new = np.delete(stoichiometry, independent_component, axis=0)
    solid_stoich_new = solid_stoichiometry
    log_ks_prime = log_ks
    if solid_stoichiometry.shape[0] != 0:
        solid_stoich_new = np.delete(solid_stoichiometry, independent_component, axis=0)
        log_ks_prime = log_ks[np.newaxis, :] - np.log10(
            new_x[:, np.newaxis] ** solid_stoichiometry[independent_component, :]
        )

    analc_new = np.delete(total_concentration, independent_component, axis=1)

    total_concentration, log_beta, log_ks, stoichiometry, solid_stoichiometry
    return new_x, analc_new, log_beta_prime, log_ks_prime, stoich_new, solid_stoich_new
