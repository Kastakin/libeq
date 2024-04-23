import numpy as np
from numpy.typing import NDArray
from .nr import newton_raphson
from .wrappers import outer_fixed_point


def _solids_solver(
    concentrations: NDArray,
    log_beta,
    log_ks,
    original_log_beta,
    original_log_ks,
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

    all_saturation_index = _compute_saturation_index(
        concentrations, log_ks, solid_stoichiometry
    )

    all_indices = set(range(concentrations.shape[0]))
    no_solids_points = np.where((all_saturation_index < 1).any(axis=1))[0].tolist()

    final_result[no_solids_points] = concentrations[no_solids_points]
    final_log_beta[no_solids_points] = log_beta[no_solids_points]
    final_log_ks[no_solids_points] = log_ks[no_solids_points]

    solids_points = all_indices - set(no_solids_points)

    for point in solids_points:
        solids_set = set()
        (
            c,
            point_log_ks,
            saturation_index,
            point_total_concentration,
            point_independent_component_activity,
        ) = _get_point_values(
            concentrations,
            log_ks,
            all_saturation_index,
            total_concentration,
            independent_component_activity,
            point,
        )

        adjust_solids = True
        newton_raphson_solver = outer_fixed_point(
            *outer_fiexd_point_params,
            independent_component_activity=point_independent_component_activity,
        )(newton_raphson)

        while adjust_solids:
            solids_set, adjust_solids = _update_solids_set(
                total_concentration, c, point_log_ks, saturation_index, solids_set
            )

            if not adjust_solids:
                break

            c, point_log_beta, point_log_ks = newton_raphson_solver(
                c,
                log_beta=original_log_beta[point],
                log_ks=original_log_ks[point],
                stoichiometry=stoichiometry,
                solid_stoichiometry=solid_stoichiometry,
                solids_idx=list(solids_set),
                total_concentration=point_total_concentration,
                max_iterations=1000,
                threshold=1e-10,
            )
            saturation_index = _compute_saturation_index(
                c, point_log_ks, solid_stoichiometry
            )

        final_result[point] = c
        final_log_beta[point] = point_log_beta
        final_log_ks[point] = point_log_ks
        final_saturation_index[point] = saturation_index

    return final_result, final_log_beta, final_log_ks, final_saturation_index


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


def _compute_saturation_index(concentrations, log_ks, solid_stoichiometry):
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


def _get_point_values(
    concentration,
    log_ks,
    saturation_index,
    total_concentration,
    independent_component_activity,
    point,
):  # -> tuple[Any, Any, Any, Any | None]:
    point_concentration = concentration[[point], :]
    point_log_ks = log_ks[[point], :]
    point_saturation_index = saturation_index[[point], :]
    point_total_concentration = total_concentration[[point], :]
    if independent_component_activity is not None:
        point_independent_component_activity = independent_component_activity[[point]]
    else:
        point_independent_component_activity = None
    return (
        point_concentration,
        point_log_ks,
        point_saturation_index,
        point_total_concentration,
        point_independent_component_activity,
    )
