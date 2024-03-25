import numpy as np
from numpy.typing import NDArray

from .utils import (
    _calculate_activity_coeff,
    _ionic,
    _check_outer_point_convergence,
    _update_constants,
)


def outer_fixed_point(
    ionic_strength_dependence: bool = False,
    charges: NDArray | None = None,
    ref_ionic_strength: NDArray | None = None,
    dbh_values: dict[str, NDArray] | None = None,
):
    def decorator(func):
        def wrapper(
            concentration,
            **kwargs,
        ):
            log_beta = kwargs.pop("log_beta")
            stoichiometry = kwargs["stoichiometry"]
            solid_stoichiometry = kwargs["solid_stoichiometry"]
            total_concentration = kwargs["total_concentration"]

            # Code to be executed before the decorated function
            initial_guess = np.full_like(total_concentration, 1e-6)
            old_ionic = _ionic(
                initial_guess, log_beta, stoichiometry, solid_stoichiometry, charges
            )
            old_activity = _calculate_activity_coeff(
                old_ionic, charges[:, stoichiometry.shape[0] :], dbh_values
            )
            iterations = 0
            while True:
                iterations += 1
                # Call the decorated function
                result, log_beta = func(
                    concentration,
                    log_beta=log_beta,
                    **kwargs,
                )

                # Code to be executed after the decorated function
                ionic = _ionic(
                    result, log_beta, stoichiometry, solid_stoichiometry, charges
                )
                activity = _calculate_activity_coeff(
                    ionic, charges[:, stoichiometry.shape[0] :], dbh_values
                )
                if _check_outer_point_convergence(activity, old_activity):
                    # print(func.__name__)
                    # print(f"Outer converged in {iterations} iterations")
                    # print("------------------------")
                    break

                old_activity = activity
                old_ionic = ionic
                log_beta = _update_constants(
                    log_beta, ionic, ref_ionic_strength, dbh_values
                )

            return result, log_beta

        if ionic_strength_dependence:
            return wrapper
        else:
            return func

    return decorator
