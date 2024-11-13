import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def species_concentration(
    concentration,
    log_beta,
    stoichiometry,
    full=False,
):
    r"""
    Calculate the species concentrations through the mass action law.

    $$
    S_{i} = \beta_i \prod_{j=1}^{N_c} C_j^{p_{ij}}
    $$

    With $S_i$ being the concentration of the species $i$, $\beta_i$ the equilibrium constant of the species $i$,
    $C_j$ the concentration of the component $j$, and $p_{ij}$ the stoichiometric coefficient of the component $j$ in the species $i$.

    Parameters
    ----------
    concentration : numpy.ndarray
        The concentration array of shape (n, c+p), where n is the number of points c is the number of components and p is the number of solid species.
    log_beta : numpy.ndarray
        The logarithm of the equilibrium constants with shape (n, s), where s is the number of solid species.
    stoichiometry : numpy.ndarray
        The stoichiometry matrix with shape (n, s), where s is the number of soluble species.
    full : bool, optional
        If True, return the concentrations of all species including the original concentrations.
        If False, return only the concentrations of the new species.

    Returns
    -------
    numpy.ndarray
        The calculated species concentrations.

    """
    nc = stoichiometry.shape[0]
    _c = np.log10(concentration[:, :nc])

    cext = 10 ** (log_beta + _c @ stoichiometry)

    if full:
        p = np.concatenate((concentration, cext), axis=1)
    else:
        p = cext

    return p
