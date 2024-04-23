from copy import deepcopy
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from libeq.data_structure import SolverData
from libeq.utils import species_concentration
from libeq.outer_fixed_point import outer_fixed_point

from .damping import damping
from .nr import newton_raphson
from .solids_solver import _solids_solver


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
    original_log_beta = data.log_beta
    original_log_ks = data.log_ks
    ionic_strength_dependence = data.ionic_strength_dependence
    charges = np.atleast_2d(np.concatenate((data.charges, data.species_charges)))
    dbh_values = deepcopy(data.dbh_values)

    # Get the total concentration values depending if mode is titration or distribution
    if mode == "titration":
        c0 = data.c0
        ct = data.ct
        v0 = data.v0
        v_add = data.v_add
        if v_add is None:
            v_add = np.arange(data.n_add) * (data.v_increment)
        total_concentration: NDArray = (
            ((c0 * v0)[:, np.newaxis] + ct[:, np.newaxis] * v_add) / (v_add + v0)
        ).T

        independent_component_activity = None

    elif mode == "distribution":
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
            stoichiometry,
            solid_stoichiometry,
            original_log_beta,
            original_log_ks,
            charges,
            total_concentration,
            independent_component_activity,
        ) = _simplify_model(
            charges,
            independent_component,
            total_concentration,
            original_log_beta,
            original_log_ks,
            stoichiometry,
            solid_stoichiometry,
        )
    else:
        raise ValueError("Invalid work mode")

    outer_fiexd_point_params = [
        ionic_strength_dependence,
        charges,
        data.reference_ionic_str_species,
        data.reference_ionic_str_solids,
        dbh_values,
    ]
    damping_fn = outer_fixed_point(
        *outer_fiexd_point_params,
        independent_component_activity=independent_component_activity,
    )(damping)

    nr_fn = outer_fixed_point(
        *outer_fiexd_point_params,
        independent_component_activity=independent_component_activity,
    )(newton_raphson)

    # Get the initial guess for the free concentrations
    initial_guess, log_beta, log_ks = damping_fn(
        np.full_like(total_concentration, 1e-10),
        log_beta=original_log_beta,
        log_ks=original_log_ks,
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
    result, log_beta, log_ks = nr_fn(
        initial_guess,
        log_beta=original_log_beta,
        log_ks=original_log_ks,
        stoichiometry=stoichiometry,
        solid_stoichiometry=solid_stoichiometry,
        total_concentration=total_concentration,
        max_iterations=1000,
        threshold=1e-10,
    )

    if data.nf > 0:
        result, log_beta, log_ks, saturation_index = _solids_solver(
            result,
            log_beta,
            log_ks,
            original_log_beta,
            original_log_ks,
            stoichiometry,
            solid_stoichiometry,
            total_concentration,
            outer_fiexd_point_params=outer_fiexd_point_params,
            independent_component_activity=independent_component_activity,
        )
    else:
        saturation_index = np.empty((result.shape[0], 0))

    if mode == "distribution":
        result, log_beta, log_ks, total_concentration = _expand_result(
            result,
            independent_component,
            independent_component_concentration,
            total_concentration,
            log_beta,
            log_ks,
            data.stoichiometry,
            data.solid_stoichiometry,
        )

    return result, log_beta, log_ks, saturation_index, total_concentration


def _expand_result(
    result,
    independent_component,
    independent_component_concentration,
    total_concentration,
    log_beta,
    log_ks,
    original_stoichiometry,
    original_solid_stoichiometry,
):
    result = np.insert(
        result, independent_component, independent_component_concentration, axis=1
    )

    log_beta = log_beta - np.log10(
        independent_component_concentration[:, np.newaxis]
        ** original_stoichiometry[independent_component, :]
    )

    if original_solid_stoichiometry.shape[0] != 0:
        log_ks = log_ks + np.log10(
            independent_component_concentration[:, np.newaxis]
            ** original_solid_stoichiometry[independent_component, :]
        )

    calculated_total_concentration = (
        species_concentration(result, log_beta, original_stoichiometry)
        * original_stoichiometry[independent_component, :]
    ).sum(axis=1) + independent_component_concentration

    total_concentration = np.insert(
        total_concentration,
        independent_component,
        calculated_total_concentration,
        axis=1,
    )

    return result, log_beta, log_ks, total_concentration


def _simplify_model(
    charges,
    independent_component,
    total_concentration,
    log_beta,
    log_ks,
    stoichiometry,
    solid_stoichiometry,
):
    (
        independent_concentration,
        total_concentration,
        log_beta,
        log_ks,
        stoichiometry,
        solid_stoichiometry,
    ) = _freeze_concentration(
        independent_component,
        total_concentration,
        log_beta,
        log_ks,
        stoichiometry,
        solid_stoichiometry,
    )

    independent_component_charge = charges[:, independent_component]
    independent_component_activity = 0.5 * (
        independent_concentration * (independent_component_charge**2)
    )
    charges = np.delete(charges, independent_component, axis=1)

    return (
        stoichiometry,
        solid_stoichiometry,
        log_beta,
        log_ks,
        charges,
        total_concentration,
        independent_component_activity,
    )


def _freeze_concentration(
    independent_component,
    total_concentration,
    log_beta,
    log_ks,
    stoichiometry,
    solid_stoichiometry,
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
    analc_new = np.delete(total_concentration, independent_component, axis=1)
    log_beta_prime = log_beta[np.newaxis, :] + np.log10(
        new_x[:, np.newaxis] ** stoichiometry[independent_component, :]
    )
    stoich_new = np.delete(stoichiometry, independent_component, axis=0)
    solid_stoich_new = solid_stoichiometry
    log_ks_prime = log_ks
    if solid_stoichiometry.shape[0] != 0:
        log_ks_prime = log_ks[np.newaxis, :] - np.log10(
            new_x[:, np.newaxis] ** solid_stoichiometry[independent_component, :]
        )
        solid_stoich_new = np.delete(solid_stoichiometry, independent_component, axis=0)

    return (
        new_x,
        analc_new,
        log_beta_prime,
        log_ks_prime,
        stoich_new,
        solid_stoich_new,
    )
