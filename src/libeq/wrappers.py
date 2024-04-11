import numpy as np
from numpy.typing import NDArray

from .species_conc import species_concentration

from .utils import (
    _calculate_activity_coeff,
    _ionic,
    _check_outer_point_convergence,
    _update_formation_constants,
)


def outer_fixed_point(
    ionic_strength_dependence: bool = False,
    charges: NDArray | None = None,
    ref_ionic_strength: NDArray | None = None,
    dbh_values: dict[str, NDArray] | None = None,
    independent_component_activity: NDArray | None = None,
):
    def decorator(func):
        def wrapper(
            concentration,
            **kwargs,
        ):
            log_beta = kwargs.pop("log_beta")
            log_ks = kwargs.pop("log_ks")
            stoichiometry = kwargs["stoichiometry"]
            # solid_stoichiometry = kwargs["stoichiometry"]
            # total_concentration = kwargs["total_concentration"]

            n_components = stoichiometry.shape[0]
            n_species = stoichiometry.shape[1]

            # Code to be executed before the decorated function
            concentrations = species_concentration(
                concentration, log_beta, stoichiometry, full=True
            )
            if independent_component_activity is not None:
                trasnsposed_activity = independent_component_activity[:, np.newaxis]
            else:
                trasnsposed_activity = None
            old_ionic = _ionic_fn(
                _select_species_concentration(concentrations, n_components, n_species),
                charges,
                independent_component_activity=trasnsposed_activity,
            )

            old_activity = _calculate_activity_coeff(
                old_ionic, charges[:, -n_species:], dbh_values
            )

            iterations = 0
            while True:
                iterations += 1
                # Call the decorated function
                result, log_beta = func(
                    concentration,
                    log_beta=log_beta,
                    log_ks=log_ks,
                    **kwargs,
                )

                # Code to be executed after the decorated function
                ionic = _ionic_fn(
                    _select_species_concentration(
                        concentrations, n_components, n_species
                    ),
                    charges,
                    independent_component_activity=trasnsposed_activity,
                )
                activity = _calculate_activity_coeff(
                    ionic, charges[:, -n_species:], dbh_values
                )
                if _check_outer_point_convergence(activity, old_activity):
                    # print(func.__name__)
                    # print(f"Outer converged in {iterations} iterations")
                    # print("------------------------")
                    break

                old_activity = activity
                old_ionic = ionic
                log_beta = _update_formation_constants(
                    log_beta, ionic, ref_ionic_strength, dbh_values
                )
                # log_ks = _update_solubility_products(
                #     log_ks, ionic, ref_ionic_strength, dbh_values
                # )

            return result, log_beta

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
            return func

    return decorator


def _select_species_concentration(c, n_components, n_species):
    return np.concatenate(
        (
            c[:, :n_components],
            c[:, -n_species:],
        ),
        axis=1,
    )
