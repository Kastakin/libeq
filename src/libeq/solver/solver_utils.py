from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from libeq.data_structure import (
    SolverData,
    SimulationTitrationParameters,
    PotentiometryTitrationsParameters,
)
from libeq.utils import species_concentration


def _assemble_outer_fixed_point_params(
    data: SolverData,
    charges,
    background_ions_concentration,
    independent_component_activity,
):
    outer_fixed_point_params = {
        "ionic_strength_dependence": data.ionic_strength_dependence,
        "reference_ionic_str_species": data.reference_ionic_str_species,
        "reference_ionic_str_solids": data.reference_ionic_str_solids,
        "dbh_values": deepcopy(data.dbh_values),
        "charges": charges,
        "independent_component_activity": independent_component_activity,
        "background_ions_concentration": background_ions_concentration,
    }
    return outer_fixed_point_params


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


def _prepare_common_data(data: SolverData):
    stoichiometry = data.stoichiometry
    solid_stoichiometry = data.solid_stoichiometry
    original_log_beta = data.log_beta
    original_log_ks = data.log_ks
    charges = np.atleast_2d(np.concatenate((data.charges, data.species_charges)))
    independent_component_activity = None

    return (
        stoichiometry,
        solid_stoichiometry,
        original_log_beta,
        original_log_ks,
        charges,
        independent_component_activity,
    )


def _prepare_distribution_data(data: SolverData):
    (
        stoichiometry,
        solid_stoichiometry,
        original_log_beta,
        original_log_ks,
        charges,
        independent_component_activity,
    ) = _prepare_common_data(data)

    independent_component = data.distribution_opts.independent_component
    independent_component_concentration = 10 ** -np.arange(
        data.distribution_opts.initial_log,
        (data.distribution_opts.final_log + data.distribution_opts.log_increments),
        data.distribution_opts.log_increments,
    )
    total_concentration = np.repeat(
        data.distribution_opts.c0[np.newaxis, :],
        len(independent_component_concentration),
        axis=0,
    )
    total_concentration[:, independent_component] = independent_component_concentration

    background_ions_concentration = data.distribution_opts.cback

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

    return (
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
    )


def _prepare_titration_data(data: SolverData):
    (
        stoichiometry,
        solid_stoichiometry,
        original_log_beta,
        original_log_ks,
        charges,
        independent_component_activity,
    ) = _prepare_common_data(data)

    total_concentration = _titration_total_c(data.titration_opts)
    original_log_beta = np.tile(original_log_beta, (total_concentration.shape[0], 1))
    original_log_ks = np.tile(original_log_ks, (total_concentration.shape[0], 1))
    background_ions_concentration = _titration_background_ions_c(data.titration_opts)

    return (
        stoichiometry,
        solid_stoichiometry,
        original_log_beta,
        original_log_ks,
        charges,
        background_ions_concentration,
        independent_component_activity,
        total_concentration,
    )


def _titration_total_c(
    titration_data: PotentiometryTitrationsParameters | SimulationTitrationParameters,
    idx=None,
):
    c0 = titration_data.c0
    ct = titration_data.ct
    v0 = titration_data.v0

    v_add = _get_titration_vadd(titration_data, idx)

    total_concentration: NDArray = (
        ((c0 * v0)[:, np.newaxis] + ct[:, np.newaxis] * v_add) / (v_add + v0)
    ).T

    return total_concentration


def _titration_background_ions_c(
    titration_data: PotentiometryTitrationsParameters | SimulationTitrationParameters,
    idx=None,
):
    c0back = titration_data.c0back
    ctback = titration_data.ctback
    v0 = titration_data.v0

    v_add = _get_titration_vadd(titration_data, idx)

    background_ions_concentration: NDArray = np.atleast_2d(
        ((c0back * v0) + ctback * v_add) / (v_add + v0)
    ).T

    return background_ions_concentration


def _get_titration_vadd(
    titration_data: PotentiometryTitrationsParameters | SimulationTitrationParameters,
    idx=None,
):
    if isinstance(titration_data, PotentiometryTitrationsParameters):
        v_add = titration_data.v_add
        if idx is not None:
            v_add = v_add[idx]
    else:
        v_add = np.arange(titration_data.n_add) * (titration_data.v_increment)
    return v_add


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
