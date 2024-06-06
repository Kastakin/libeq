import numpy as np

from libeq.solver import _compute_saturation_index


def uncertanties(
    concentrations,
    stoichiometry,
    solid_stoichiometry,
    log_b,
    log_ks,
    log_beta_sigma,
    log_ks_sigma,
    conc_sigma,
    indepenent_comp: int | None = None,
):
    """
    Calculate the uncertainties for the components and species given the input.

    Args:
        concentrations (ndarray): Array of shape (n, m) representing the concentrations of the components and species.
        stoichiometry (ndarray): Array of shape (n, p) representing the stoichiometric coefficients of the components and species.
        solid_stoichiometry (ndarray): Array of shape (p, q) representing the stoichiometric coefficients of the solid species.
        log_b (ndarray): Array of shape (n,) representing the logarithm of the beta values.
        log_ks (ndarray): Array of shape (n,) representing the logarithm of the equilibrium constants.
        beta_sigma (float): Standard deviation of the beta values.
        ks_sigma (float): Standard deviation of the equilibrium constants.
        conc_sigma (ndarray): Array of shape (n,) representing the standard deviation of the concentrations.
        indepenent_comp (int | None, optional): Index of the independent component. Defaults to None.

    Returns:
        tuple: A tuple containing the uncertainties for the species and solid species.
    """
    # Get betas from log betas
    nc = stoichiometry.shape[0]
    ns = stoichiometry.shape[1]
    nf = solid_stoichiometry.shape[1]
    num_points = concentrations.shape[0]
    species_sigma_result = np.zeros((num_points, nc + ns))
    solid_sigma_result = np.zeros((num_points, nf))
    distribution = indepenent_comp is not None
    all_beta_sigma = log_beta_sigma * np.log(10) * (10**log_b)
    all_ks_sigma = log_ks_sigma * np.log(10) * (10**log_ks)

    for point in range(num_points):
        beta = 10 ** log_b[point]
        ks = 10 ** log_ks[point]
        beta_sigma = all_beta_sigma[point]
        ks_sigma = all_ks_sigma[point]

        c_free = concentrations[point, :nc]
        c_spec = concentrations[point, nc + nf :]
        c_solid = concentrations[point, nc : nc + nf]
        # c_free, c_spec, c_solid = np.atleast_2d(c_free, c_spec, c_solid)

        saturation_index = _compute_saturation_index(
            c_free[np.newaxis, :], log_ks[point], solid_stoichiometry
        )[0]

        with_solids = any(c_solid > 0)

        to_skip = np.concatenate(([False for _ in range(nc)], c_solid == 0))
        if with_solids:
            nt = nc + nf
        else:
            nt = nc

        # Define dimension of arrays required
        M = np.zeros(shape=(nt, nt))

        der_free_beta = np.zeros(shape=(nc, ns))
        der_free_tot = np.zeros(shape=(nc, nc))
        der_free_ks = np.zeros(shape=(nc, nf))

        der_solid_beta = np.zeros(shape=(nf, ns))
        der_solid_tot = np.zeros(shape=(nf, nc))
        der_solid_ks = np.zeros(shape=(nf, nf))

        b = -stoichiometry * (c_spec / beta)
        d = np.identity(nc)
        f = np.zeros(shape=(nc, nf))

        # Compute common matrix term
        M[:nc, :nc] = (
            (
                np.tile(c_spec, (nc, nc, 1))
                / np.tile(c_free.reshape((nc, 1)), (nc, 1, ns))
            )
            * np.tile(stoichiometry, (nc, 1, 1))
            * np.rot90(np.tile(stoichiometry, (nc, 1, 1)), -1, axes=(0, 1))
        ).sum(axis=-1)
        M[:nc, :nc] += d

        if with_solids:
            M[:nc, nc:nt] = solid_stoichiometry
            M[nc:nt, :nc] = solid_stoichiometry.T * (
                np.tile(saturation_index, (nc, 1)).T / np.tile(c_free, (nt - nc, 1))
            )

            f = np.concatenate((f, np.diag(saturation_index / ks)), axis=0)
            b = np.concatenate(
                (
                    b,
                    [[0 for _ in range(ns)] for _ in range(nf)],
                )
            )
            d = np.concatenate(
                (
                    d,
                    [[0 for _ in range(nc)] for _ in range(nf)],
                )
            )

            der_solid_beta = np.delete(der_solid_beta, c_solid == 0, axis=0)
            der_solid_tot = np.delete(der_solid_tot, c_solid == 0, axis=0)
            der_solid_ks = np.delete(der_solid_ks, c_solid == 0, axis=0)

            M = np.delete(M, to_skip, axis=0)
            M = np.delete(M, to_skip, axis=1)

            b = np.delete(b, to_skip, axis=0)

            d = np.delete(d, to_skip, axis=0)

            f = np.delete(f, to_skip, axis=0)

        if indepenent_comp is not None:
            M = np.delete(M, indepenent_comp, axis=0)
            M = np.delete(M, indepenent_comp, axis=1)

            b = np.delete(b, indepenent_comp, axis=0)
            d = np.delete(d, indepenent_comp, axis=0)
            f = np.delete(f, indepenent_comp, axis=0)

            der_free_beta = np.delete(der_free_beta, indepenent_comp, 0)
            der_free_tot = np.delete(der_free_tot, indepenent_comp, 0)
            der_free_ks = np.delete(der_free_ks, indepenent_comp, 0)

        # Solve the systems of equations
        for i in range(ns):
            solution = np.linalg.solve(M, b[:, i])
            der_free_beta[:, i] = solution[: (nc - 1 if distribution else nc)]
            if with_solids:
                der_solid_beta[:, i] = solution[(nc - 1 if distribution else nc) :]

        for r in range(nc):
            solution = np.linalg.solve(M, d[:, r])
            der_free_tot[:, r] = solution[: (nc - 1 if distribution else nc)]
            if with_solids:
                der_solid_tot[:, r] = solution[(nc - 1 if distribution else nc) :]

        if with_solids:
            for k, skip in enumerate(to_skip[-nf:]):
                if skip:
                    continue
                solution = np.linalg.solve(M, f[:, k])
                der_free_ks[:, k] = solution[: (nc - 1 if distribution else nc)]
                der_solid_ks[:, k] = solution[(nc - 1 if distribution else nc) :]

        if with_solids:
            null_solids_index = np.nonzero(c_solid == 0)[0]
            if null_solids_index.size:
                der_solid_beta = np.insert(der_solid_beta, null_solids_index, 0, axis=0)
                der_solid_tot = np.insert(der_solid_tot, null_solids_index, 0, axis=0)
                der_solid_ks = np.insert(der_solid_ks, null_solids_index, 0, axis=0)

        if distribution:
            der_free_beta = np.insert(der_free_beta, indepenent_comp, 0, axis=0)
            der_free_tot = np.insert(der_free_tot, indepenent_comp, 0, axis=0)
            der_free_ks = np.insert(der_free_ks, indepenent_comp, 0, axis=0)

        # Compute derivatives for the species
        der_spec_beta = (
            np.rot90(np.tile(stoichiometry.T, (ns, 1, 1)), -1)
            * (
                np.stack([np.tile(c_spec, (ns, 1)).T for _ in range(nc)], axis=-1)
                / c_free
            )
            * np.tile(der_free_beta.T, (ns, 1, 1))
        ).sum(axis=-1) + np.diag(c_spec / beta)

        der_spec_tot = (
            np.rot90(np.tile(stoichiometry.T, (nc, 1, 1)), -1)
            * (
                np.stack([np.tile(c_spec, (nc, 1)).T for _ in range(nc)], axis=-1)
                / c_free
            )
            * np.tile(der_free_tot.T, (ns, 1, 1))
        ).sum(axis=-1)

        der_spec_ks = (
            np.rot90(np.tile(stoichiometry.T, (nf, 1, 1)), -1)
            * (
                np.stack([np.tile(c_spec, (nf, 1)).T for _ in range(nc)], axis=-1)
                / c_free
            )
            * np.tile(der_free_ks.T, (ns, 1, 1))
        ).sum(axis=-1)

        # Calculate uncertanity for components and species given the input
        comp_sigma = np.sqrt(
            ((der_free_beta**2) * (beta_sigma**2)).sum(axis=1)
            + ((der_free_tot**2) * (conc_sigma[point] ** 2)).sum(axis=1)
            + ((der_free_ks**2) * (ks_sigma**2)).sum(axis=1)
        )

        species_sigma = np.sqrt(
            ((der_spec_beta**2) * (beta_sigma**2)).sum(axis=1)
            + ((der_spec_tot**2) * (conc_sigma[point] ** 2)).sum(axis=1)
            + ((der_spec_ks**2) * (ks_sigma**2)).sum(axis=1)
        )
        species_sigma = np.concatenate((comp_sigma, species_sigma))

        if with_solids:
            solid_sigma = np.sqrt(
                ((der_solid_beta**2) * (beta_sigma**2)).sum(axis=1)
                + ((der_solid_tot**2) * (conc_sigma[point] ** 2)).sum(axis=1)
                + ((der_solid_ks**2) * (ks_sigma**2)).sum(axis=1)
            )
        else:
            solid_sigma = np.zeros(shape=nf)

        species_sigma_result[point] = species_sigma
        solid_sigma_result[point] = solid_sigma

    return species_sigma_result, solid_sigma_result
