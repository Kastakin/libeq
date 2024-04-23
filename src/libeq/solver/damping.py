import numpy as np
import numpy.typing as npt

from libeq.utils import species_concentration


def pcfm(
    concentration,
    *,
    log_beta,
    stoichiometry,
    total_concentration,
    max_iterations=1000,
    tol=2.5e-1,
    **kwargs,
) -> npt.NDArray:
    """
    Solve the mass balance equations using the Positive Continious Fraction Method.

    Parameters
    ----------
    concentration : numpy.ndarray
        The concentration array of shape (n, c), where n is the number of points c is the number of components.
    log_beta : numpy.ndarray
        The logarithm of the equilibrium constants with shape (n, s), where s is the number of soluble species.
    stoichiometry : numpy.ndarray
        The stoichiometry matrix with shape (n, s), where s is the number of soluble species.
    total_concentration : numpy.ndarray
        The total concentration vector with shape (n, c), where n is the number of points c is the number of components..
    max_iterations : int, optional
        The maximum number of iterations (default is 1000).
    tol : float, optional
        The tolerance for convergence criteria (default is 2.5e-1).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    c_spec : numpy.ndarray
        The solution to the mass balance equations.
    """
    nc = stoichiometry.shape[0]

    coeff = np.array([0 for _ in range(nc)])
    exponent = 1 / np.max(
        np.where(stoichiometry == 0, 1, np.abs(stoichiometry)), axis=1
    )
    full_stoichiometry = np.concatenate((np.eye(nc), stoichiometry), axis=1)

    pstoich, nstoich = _pos_neg_stoich(full_stoichiometry)

    iteration = 0
    while True:
        c_spec = species_concentration(
            concentration,
            log_beta,
            stoichiometry,
            full=True,
        )

        sum_reac, sum_prod = _sumps(c_spec, total_concentration, pstoich, nstoich)

        conv_criteria = np.abs((sum_reac - sum_prod) / (sum_reac + sum_prod))

        if np.all(conv_criteria <= tol) or iteration >= max_iterations:
            return c_spec[:, :nc]

        ratio = sum_prod / sum_reac
        new_coeff = 0.9 - np.where(ratio < 1.0, ratio, 1 / ratio) * 0.8

        if iteration == 0:
            coeff = new_coeff
        coeff = np.where(new_coeff > coeff, new_coeff, coeff)

        concentration *= coeff * ratio ** (exponent) + (1 - coeff)

        iteration += 1


def _pos_neg_stoich(full_stoichiometry):
    pstoich = np.zeros_like(full_stoichiometry)
    nstoich = np.zeros_like(full_stoichiometry)

    pos = full_stoichiometry >= 0
    pstoich[pos] = full_stoichiometry[pos]
    nstoich[~pos] = np.abs(full_stoichiometry[~pos])
    return pstoich, nstoich


def _sumps(species, analyticalc, pstoich, nstoich):
    sumrp = np.sum(species[:, np.newaxis, :] * pstoich[np.newaxis, ...], axis=2)
    sumrn = np.abs(analyticalc) + sumrp
    sumpn = np.sum(species[:, np.newaxis, :] * nstoich[np.newaxis, ...], axis=2)
    sumpp = analyticalc + sumpn

    tpos = analyticalc >= 0.0
    sumr = np.where(tpos, sumrp, sumrn)
    sump = np.where(tpos, sumpp, sumpn)
    return sumr, sump
