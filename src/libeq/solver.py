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

    # Get the total concentration values depending if mode is titration or distribution
    if mode == "titration":
        c0 = data.c0
        ct = data.ct
        v0 = data.v0
        v_add = data.v_add
        total_concentration: NDArray = (
            ((c0 * v0)[:, np.newaxis] + ct[:, np.newaxis] * v_add) / (v_add + v0)
        ).T

    elif mode == "distribution":
        total_concentration = data.c0_tot
        independent_component = data.independent_component

        # Reduce the problem by accounting for the independent component
        total_concentration, log_beta, log_ks, stoichiometry, solid_stoichiometry = (
            freeze_concentration(
                independent_component,
                total_concentration,
                log_beta,
                log_ks,
                stoichiometry,
                solid_stoichiometry,
                ionic_strength_dependence=ionic_strength_dependence,
            )
        )

    solids_idx = []
    charges = np.atleast_2d(np.concatenate((data.charges, data.species_charges)))

    damping_fn = outer_fixed_point(
        data.ionic_strength_dependence,
        charges,
        data.ref_ionic_str,
        data.dbh_values,
    )(damping)

    nr_fn = outer_fixed_point(
        data.ionic_strength_dependence,
        charges,
        data.ref_ionic_str,
        data.dbh_values,
    )(NewtonRaphson)

    # Get the initial guess for the free concentrations
    initial_guess, log_beta = damping_fn(
        np.full_like(total_concentration, 1e-6),
        log_beta=log_beta,
        stoichiometry=stoichiometry,
        solid_stoichiometry=solid_stoichiometry,
        total_concentration=total_concentration,
        tol=1e-3,
    )

    # Apply Newton-Raphson iterations
    result, log_beta = nr_fn(
        initial_guess,
        log_beta=log_beta,
        log_ks=log_ks,
        stoichiometry=stoichiometry,
        solid_stoichiometry=solid_stoichiometry,
        total_concentration=total_concentration,
        solids_idx=solids_idx,
        max_iterations=1000,
        threshold=1e-10,
    )

    print(log_beta)

    return result, log_beta


def saturation_index():
    pass


def freeze_concentration():
    pass
