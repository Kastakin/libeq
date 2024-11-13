from typing import Literal

import numpy as np

from libeq.data_structure import SolverData
from libeq.outer_fixed_point import outer_fixed_point
from ..utils import species_concentration

from .damping import pcf
from .nr import newton_raphson
from .solids_solver import solids_solver
from .solver_utils import (
    _assemble_outer_fixed_point_params,
    _expand_result,
    _prepare_distribution_data,
    _prepare_titration_data,
)


def EqSolver(
    data: SolverData, mode: Literal["titration", "distribution"] = "titration"
):
    """
    Solve the equilibrium equations for a given chemical system.

    The solver uses a conjuntion of methods to solve the problem at hand. In particular:

    - The Positive Continious Fraction Method (PCFM) is used to presolve the equilibrium equations.
    - The Newton-Raphson method is used to solve the equilibrium equations.
    - The outer fixed point method is used to solve the equilibrium equations for solids.

    Parameters
    ----------
    data : SolverData
        The input data containing all the necessary information for the solver.
    mode : {"titration", "distribution"}, optional
        The mode of operation for the solver. Default is "titration".

    Returns
    -------
    result : ndarray
        The calculated equilibrium concentrations.
    log_beta : ndarray
        The logarithm of the stability constants.
    log_ks : ndarray
        The logarithm of the solubility products.
    saturation_index : ndarray
        The calculated saturation indices for solid phases.
    total_concentration : ndarray
        The total concentrations used in the calculations.
    """
    # Get the total concentration values depending if mode is titration or distribution
    if mode == "titration":
        (
            stoichiometry,
            solid_stoichiometry,
            original_log_beta,
            original_log_ks,
            charges,
            background_ions_concentration,
            independent_component_activity,
            total_concentration,
        ) = _prepare_titration_data(data)
    elif mode == "distribution":
        (
            stoichiometry,
            solid_stoichiometry,
            original_log_beta,
            original_log_ks,
            charges,
            background_ions_concentration,
            independent_component_activity,
            total_concentration,
            independent_component,
            independent_component_concentration,
        ) = _prepare_distribution_data(data)
    else:
        raise ValueError("Invalid work mode")

    outer_fiexd_point_params = _assemble_outer_fixed_point_params(
        data, charges, background_ions_concentration, independent_component_activity
    )

    # Solve the equilibrium equations
    result, log_beta, log_ks, saturation_index, total_concentration = (
        solve_equilibrium_equations(
            stoichiometry=stoichiometry,
            solid_stoichiometry=solid_stoichiometry,
            original_log_beta=original_log_beta,
            original_log_ks=original_log_ks,
            total_concentration=total_concentration,
            outer_fiexd_point_params=outer_fiexd_point_params,
        )
    )

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


def solve_equilibrium_equations(
    *,
    stoichiometry,
    solid_stoichiometry,
    original_log_beta,
    original_log_ks,
    total_concentration,
    outer_fiexd_point_params,
    initial_guess=None,
    full=False,
):
    if initial_guess is None:
        initial_guess = np.full_like(total_concentration, 1e-10)
    else:
        initial_guess = np.atleast_2d(initial_guess)

    damping_fn = outer_fixed_point(
        **outer_fiexd_point_params,
    )(pcf)

    nr_fn = outer_fixed_point(
        **outer_fiexd_point_params,
    )(newton_raphson)

    # Get the initial guess for the free concentrations
    damped_guess, log_beta, log_ks = damping_fn(
        initial_guess,
        log_beta=original_log_beta,
        log_ks=original_log_ks,
        stoichiometry=stoichiometry,
        solid_stoichiometry=solid_stoichiometry,
        total_concentration=total_concentration,
        tol=1e-3,
    )

    # Add the solid concentrations to the initial guess
    damped_guess = np.concatenate(
        (
            damped_guess,
            np.full((damped_guess.shape[0], solid_stoichiometry.shape[1]), 0),
        ),
        axis=1,
    )

    # Apply Newton-Raphson iterations
    result, log_beta, log_ks = nr_fn(
        damped_guess,
        log_beta=original_log_beta,
        log_ks=original_log_ks,
        stoichiometry=stoichiometry,
        solid_stoichiometry=solid_stoichiometry,
        total_concentration=total_concentration,
        max_iterations=1000,
        threshold=1e-10,
    )

    if solid_stoichiometry.shape[1] > 0:
        result, log_beta, log_ks, saturation_index = solids_solver(
            result,
            log_beta,
            log_ks,
            original_log_beta,
            original_log_ks,
            stoichiometry,
            solid_stoichiometry,
            total_concentration,
            outer_fiexd_point_params=outer_fiexd_point_params,
        )
    else:
        saturation_index = np.empty((result.shape[0], 0))

    if full:
        result = species_concentration(result, log_beta, stoichiometry, full)

    return result, log_beta, log_ks, saturation_index, total_concentration
