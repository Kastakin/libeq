import numpy as np
from numpy.typing import NDArray


def _ionic(concentration: NDArray, charges: NDArray, **kwargs) -> NDArray:
    return 0.5 * (concentration * (charges**2)).sum(axis=1, keepdims=True)


def _calculate_activity_coeff(ionic_strength, charges, dbh_values):
    return (
        -dbh_values["adh"]
        * (charges**2)
        * (
            np.sqrt(ionic_strength)
            / (1 + (dbh_values["bdh"] * np.sqrt(ionic_strength)))
        )
        + dbh_values["cdh"] * ionic_strength
        + dbh_values["ddh"] * ionic_strength**1.5
        + dbh_values["edh"] * ionic_strength**2
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


def _check_outer_point_convergence(log_beta, old_log_beta, log_ks, old_log_ks):
    return np.all((np.abs(log_beta - old_log_beta) < 1e-4)) and np.all(
        np.abs(log_ks - old_log_ks) < 1e-4
    )
