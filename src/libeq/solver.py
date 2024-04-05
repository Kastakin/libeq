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
        if data.ionic_strength_dependence:
            print(
                "Ionic strength dependence is not implemented for distribution mode!\n No ionic strength dependence will be considered."
            )
            data.ionic_strength_dependence = False
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

    return result, log_beta


def saturation_index():
    pass


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
        log_ks_prime = (10 ** log_ks[np.newaxis, :]) / (
            new_x[:, np.newaxis] ** solid_stoichiometry[independent_component, :]
        )

    analc_new = np.delete(total_concentration, independent_component, axis=1)

    total_concentration, log_beta, log_ks, stoichiometry, solid_stoichiometry
    return analc_new, log_beta_prime, log_ks_prime, stoich_new, solid_stoich_new
