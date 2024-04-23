import numpy as np


def species_concentration(
    concentration,
    log_beta,
    stoichiometry,
    full=False,
):
    r"""Compute the free concentrations for the extended components.

    The concentration of the complexes is calculated by means of the
    equilibrium condition.

    .. math::`c_{i+S} = \beta_i \prod_{j=1}^E c_j^p_{ji}`

    This is an auxiliary function.

    Parameters:
        concentration (:class:`numpy.ndarray`): The free concentrations of the
            free components. It must be an (*S*,) or (*N*, *S*)-sized array
            where *S* is the number of free components. This parameter can be
            a masked array. In this case, the return concentration matrix will
            also be masked.
        beta (:class:`numpy.ndarray`): The equilibrium constants. The last
            dimmension must be E-sized and the rest of the dimmensions must be
            compatible with those of **concentration**.
        stoichiometry (:class:`numpy.ndarray`): The stoichiometric coefficient
            matrix. It must be (*E*, *S*)-sized where E is the number of
            equilibria.
        full (bool): If set, the return array will be the full (*N*, *S* + *E*)
            array. If unset only the extra calculated array (*N*, *E*) will be
            returned.
    Returns:
        :class:`numpy.ndarray`: array of size (*N*, *E*) containing the
            extended concentrations

    Raises:
        ValueError: If any parameter is incorrect.
    """
    nc = stoichiometry.shape[0]
    # concentration[concentration <= 0] = sys.float_info.min
    _c = np.log10(concentration[:, :nc])

    cext = 10 ** (log_beta + _c @ stoichiometry)

    if full:
        p = np.concatenate((concentration, cext), axis=1)
    else:
        p = cext

    return p
