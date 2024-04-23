import numpy as np
from numpy.typing import NDArray

from .species_conc import species_concentration

from .utils import (
    _ionic,
    _check_outer_point_convergence,
    _update_formation_constants,
    _update_solubility_products,
)


def outer_fixed_point(
    ionic_strength_dependence: bool = False,
    charges: NDArray | None = None,
    ref_ionic_strength_species: NDArray | None = None,
    ref_ionic_strength_solids: NDArray | None = None,
    dbh_values: dict[str, dict[str, NDArray]] | None = None,
    independent_component_activity: NDArray | None = None,
):
    def decorator(func):
        def unwrapped(concentration, **kwargs):
            log_beta = kwargs.pop("log_beta")
            log_ks = kwargs.pop("log_ks")
            return (
                func(concentration, log_beta=log_beta, log_ks=log_ks, **kwargs),
                log_beta,
                log_ks,
            )

        def wrapper(
            concentration,
            **kwargs,
        ):
            if independent_component_activity is not None:
                transposed_activity = independent_component_activity[:, np.newaxis]
            else:
                transposed_activity = None

            log_beta = kwargs.pop("log_beta")
            log_ks = kwargs.pop("log_ks")
            og_log_beta = log_beta.copy()
            og_log_ks = log_ks.copy()
            stoichiometry = kwargs["stoichiometry"]

            n_components = stoichiometry.shape[0]
            n_species = stoichiometry.shape[1]

            concentrations = species_concentration(
                concentration, log_beta, stoichiometry, full=True
            )
            ionic = _ionic_fn(
                _select_species_concentration(concentrations, n_components, n_species),
                charges,
                independent_component_activity=transposed_activity,
            )
            log_beta = _update_formation_constants(
                og_log_beta,
                ionic,
                ref_ionic_strength_species,
                dbh_values["species"],
            )
            log_ks = _update_solubility_products(
                og_log_ks, ionic, ref_ionic_strength_solids, dbh_values["solids"]
            )

            result = concentration
            iterations = 0
            while True:
                iterations += 1
                # Call the decorated function
                result = func(
                    result,
                    log_beta=log_beta,
                    log_ks=log_ks,
                    **kwargs,
                )

                # Code to be executed after the decorated function
                concentrations = species_concentration(
                    result, log_beta, stoichiometry, full=True
                )
                ionic = _ionic_fn(
                    _select_species_concentration(
                        concentrations, n_components, n_species
                    ),
                    charges,
                    independent_component_activity=transposed_activity,
                )

                old_log_beta = log_beta
                old_log_ks = log_ks

                log_beta = _update_formation_constants(
                    og_log_beta,
                    ionic,
                    ref_ionic_strength_species,
                    dbh_values["species"],
                )
                log_ks = _update_solubility_products(
                    og_log_ks, ionic, ref_ionic_strength_solids, dbh_values["solids"]
                )

                if _check_outer_point_convergence(
                    log_beta, old_log_beta, log_ks, old_log_ks
                ):
                    # print(func.__name__)
                    # print(f"Outer converged in {iterations} iterations")
                    # print("------------------------")
                    break
            return result, old_log_beta, old_log_ks

        def _distribution_ionic(
            concentration: NDArray,
            charges: NDArray,
            *,
            independent_component_activity: NDArray,
        ) -> NDArray:
            return _ionic(concentration, charges) + independent_component_activity

        if ionic_strength_dependence:
            if independent_component_activity is None:
                _ionic_fn = _ionic
            else:
                _ionic_fn = _distribution_ionic

            return wrapper
        else:
            return unwrapped

    return decorator


def _select_species_concentration(c, n_components, n_species):
    return np.concatenate(
        (
            c[:, :n_components],
            c[:, -n_species:],
        ),
        axis=1,
    )
