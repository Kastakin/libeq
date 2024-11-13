import numpy as np
import warnings
from libeq.excepts import DivergedIonicStrengthWarning
from numpy.typing import NDArray

from libeq.utils import species_concentration


def outer_fixed_point(
    *,
    ionic_strength_dependence: bool = False,
    charges: NDArray | None = None,
    background_ions_concentration: NDArray | float | None = None,
    reference_ionic_str_species: NDArray | None = None,
    reference_ionic_str_solids: NDArray | None = None,
    dbh_values: dict[str, dict[str, NDArray]] | None = None,
    independent_component_activity: NDArray | None = None,
):
    """
    Decorator for implementing the outer fixed point iteration method as developed described by Carrayrou, *et al.* [^1].

    Given a function capable of solving the equilibrium equations for a given set of species concentrations,
    the outer fixed point method fixes the activity coefficients and solve the equilibrium equations.
    After convergence is achieved for the species concentrations, the activity coefficients are updated and the process is repeated.
    If the activity coefficient do not change significantly between iterations, the process is considered converged and both species
    concentrations and activity coefficients are returned.

    [^1]: Carrayrou, *et al.*: *AIChE Journal* 2022, **68**, e17391

    Parameters
    ----------
    ionic_strength_dependence : bool, optional
        Flag indicating whether to consider ionic strength dependence, by default False.
    charges : numpy.ndarray or None, optional
        Array of charges for each species, by default None.
    reference_ionic_str_species : numpy.ndarray or None, optional
        Array of reference ionic strength for each species, by default None.
    reference_ionic_str_solids : numpy.ndarray or None, optional
        Array of reference ionic strength for each solid, by default None.
    dbh_values : dict[str, dict[str, numpy.ndarray]] or None, optional
        Dictionary containing the debye huckel parameters used for the species and solids, by default None.
    independent_component_activity : numpy.ndarray or None, optional
        Array of independent component activities, by default None.

    Returns
    -------
    decorator : function
        The outer fixed point decorator.

    Examples
    --------
    Apply the outer_fixed_point decorator to a function:

    ```python
    @outer_fixed_point(
        ionic_strength_dependence=True,
        charges=[1,0,-1],
        ref_ionic_strength_species=[0.1,0.1,0.1],
        ref_ionic_strength_solids=[0.1,0.1,0.1],
        dbh_values={"species": {...}, "solids": {...}
        independent_component_activity=None
        )
    def equilibrium_equations(concentration, log_beta, log_ks):
        # implementation of equilibrium equations
        pass
    ```

    The resulting function can then be called with to compute the species concentrations at equilibrium:

    ```python
    result, final_log_beta, final_log_ks = equilibrium_equations(concentration, log_beta, log_ks)
    ```

    If ionic_strength_dependence is set to True, the decorator will perform the outer fixed point iteration method,
    otherwise it will return the result of the decorated function.

    """

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
                background_ions_concentration,
                independent_component_activity=transposed_activity,
            )
            log_beta = _update_formation_constants(
                og_log_beta,
                ionic,
                reference_ionic_str_species,
                dbh_values["species"],
            )
            log_ks = _update_solubility_products(
                og_log_ks, ionic, reference_ionic_str_solids, dbh_values["solids"]
            )

            result = concentration
            iterations = 0
            previous_convergence = np.inf

            best_log_beta = log_beta
            best_log_ks = log_ks
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
                    background_ions_concentration,
                    independent_component_activity=transposed_activity,
                )

                old_log_beta = log_beta
                old_log_ks = log_ks

                log_beta = _update_formation_constants(
                    og_log_beta,
                    ionic,
                    reference_ionic_str_species,
                    dbh_values["species"],
                )
                log_ks = _update_solubility_products(
                    og_log_ks, ionic, reference_ionic_str_solids, dbh_values["solids"]
                )

                converged, previous_convergence, is_best = (
                    _check_outer_point_convergence(
                        log_beta, old_log_beta, log_ks, old_log_ks, previous_convergence
                    )
                )

                if converged:
                    break
                elif is_best:
                    best_log_beta = log_beta
                    best_log_ks = log_ks

                if iterations > 1000 or np.isnan(previous_convergence):
                    warning = DivergedIonicStrengthWarning(
                        msg="Divergence in outer fixed point iteration",
                        last_value=ionic,
                    )
                    warnings.warn(warning)
                    old_log_beta = best_log_beta
                    old_log_ks = best_log_ks
                    break
            return result, old_log_beta, old_log_ks

        def _distribution_ionic(
            concentration: NDArray,
            charges: NDArray,
            background_ions_concentration: NDArray | float | None,
            *,
            independent_component_activity: NDArray,
        ) -> NDArray:
            return (
                _ionic(concentration, charges, background_ions_concentration)
                + independent_component_activity
            )

        def _ionic(
            concentration: NDArray,
            charges: NDArray,
            background_ions_concentration: NDArray,
            **kwargs,
        ) -> NDArray:
            return 0.5 * (
                (concentration * (charges**2)).sum(axis=1, keepdims=True)
                + background_ions_concentration
            )

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


def _update_formation_constants(
    log_beta, ionic_strength, ref_ionic_strength, dbh_values
):
    cis = np.tile(ionic_strength, ref_ionic_strength.shape[0])
    radqcis = np.sqrt(cis)
    fib2 = radqcis / (1 + (dbh_values["bdh"] * radqcis))
    return (
        log_beta
        - dbh_values["azast"] * (fib2 - dbh_values["fib"])
        + dbh_values["cdh"] * (cis - ref_ionic_strength)
        + dbh_values["ddh"]
        * ((cis * radqcis) - (ref_ionic_strength * (ref_ionic_strength) ** 0.5))
        + dbh_values["edh"] * ((cis**2) - (ref_ionic_strength**2))
    )


def _update_solubility_products(log_ks, ionic_strength, ref_ionic_strength, dbh_values):
    cis = np.tile(ionic_strength, ref_ionic_strength.shape[0])
    radqcis = np.sqrt(cis)
    fib2 = radqcis / (1 + (dbh_values["bdh"] * radqcis))
    return (
        log_ks
        + dbh_values["azast"] * (fib2 - dbh_values["fib"])
        - dbh_values["cdh"] * (cis - ref_ionic_strength)
        - dbh_values["ddh"]
        * ((cis * radqcis) - (ref_ionic_strength * (ref_ionic_strength**0.5)))
        - dbh_values["edh"] * ((cis**2) - (ref_ionic_strength**2))
    )


def _check_outer_point_convergence(
    log_beta, old_log_beta, log_ks, old_log_ks, previous_convergence
):
    soluble_diff = np.abs(log_beta - old_log_beta)
    solid_diff = np.abs(log_ks - old_log_ks)
    convergence = soluble_diff.mean()
    if solid_diff.size > 0:
        convergence += solid_diff.mean()

    best_result = previous_convergence > convergence

    converged = np.all(soluble_diff < 1e-4) and np.all(solid_diff < 1e-4)
    return converged, convergence, best_result
