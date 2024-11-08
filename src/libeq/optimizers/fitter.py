import numpy as np

from libeq.excepts import TooManyIterations


def levenberg_marquardt(x0, y, f, free_conc, jacobian, weights, capping=None, **kwargs):
    r"""Non linear fitting by means of the Levenberg-Marquardt method.

    Parameters:
        x0 (:class:`numpy.ndarray`): initial guess.
        y (:class:`numpy.ndarray`): the experimental magnitude to be fitted.
            If must be a 1D-shaped array.
        weights (1D-array of floats): containing the values for weighting. It
            must be the same shape and type as *y*.
        f (callable): A function that accepts the values of *x0* as well as
            the free concentrations and return the calculated values for *y*.
            The returned array must be the same shape and type as *y*.
        free_conc (callable): A function that accepts *x0* and returns the
            values of the free concentration.
        jacobian (callable): A function that accepts *x0* and the free
            concentration array and returns the jacobian matrix.
        max_iterations (int, optional): maximum number of iterations allowed
        threshold (float, optional): criteria for convergence
        out_chisq (list, optional): If provided, the successive values for
            χ² in each iteration will be stored.
        verbosity (int, optional): An 0-2 number indicating the level of
            verbosity to be printed. 0 for mute, 1 for normal and 2 for
            pedantic output.
        report (callable, optional): A callable function that accepts the
            values of x0, iteration counter, free concentration values,
            etc., and is called every iteration in order to report on the
            progress of the fitting.
        one_iter (bool, optional): Performs one iterations and returns the
            result.
        quiet_maxits (bool, optional): Prevents this funcyion from throwing
            :class:`excepts.TooManyIterations` and quietly exits and returns
            the result when the maximum number of iterations is reached.

    Returns:
        tuple:
        - :class:`numpy.ndarray`: The refined constants in natural logarithmic
            units
        - :class:`numpy.ndarray`: The free concentrations
        - dict: Extra optional parameters

    Raises:
        ValueError: If invalid parameters are passed.
    """

    def _report(*kws):
        if report is not None:
            report(*kws)

    report = kwargs.get("report", None)
    one_iter = kwargs.get("one_iter", False)
    threshold = kwargs.pop("threshold", 1e-5)
    max_iterations = kwargs.pop("max_iterations", 100)
    quiet_maxits = kwargs.get("quiet_maxits", False)
    damping = kwargs.pop("damping", 0)
    fcapping = trivial_capping if capping is None else capping

    n_points = len(y)
    n_vars = len(x0)
    sigma_hist = []

    # import pudb
    # pudb.set_trace()

    iterations = 1
    W = np.diag(weights)
    # assert W.shape == (n_points, n_points)

    x = np.copy(x0)
    concs = free_conc(x, iterations)
    # assert len(concs) == n_points
    y_calc = f(concs)
    # assert y_calc.shape == y.shape

    # compute χ₂(dx)
    prev_resid = y - y_calc
    resid = y - y_calc
    sigma = fit_sigma(resid, weights, n_points, n_vars)
    while True:
        J = jacobian(concs)
        M = np.atleast_2d(np.dot(np.dot(J.T, W), J))
        D = np.diag(np.diag(M))
        V = np.atleast_1d(np.dot(np.dot(J.T, W), resid))

        try:
            dx = np.linalg.solve(M + damping * D, V)
        except np.linalg.linalg.LinAlgError:
            damping *= 10
            continue

        new_x = fcapping(x, dx)
        new_concs = free_conc(new_x, iterations)
        y_calc = f(new_concs)
        resid = y - y_calc

        sigma = fit_sigma(resid, weights, n_points, n_vars)
        old_sum_squares = weighted_sum_squares(prev_resid, weights)
        sum_squares = weighted_sum_squares(resid, weights)
        diff_ss = old_sum_squares - sum_squares

        _report(iterations, x / 2.303, dx / 2.303, sigma)

        if one_iter:
            break

        std_dev = np.diag(np.linalg.inv(M))
        beta_sigma = np.sqrt(np.abs(std_dev)) * sigma

        if (
            np.all(np.abs(dx / beta_sigma) < 0.1)
            or np.abs(diff_ss / sum_squares) < threshold
        ):
            break
        else:
            predicted_diff_ss = 2 * dx.T @ V - dx.T @ (M) @ dx
            reduction_ratio = diff_ss / predicted_diff_ss

            if reduction_ratio < 0.25:
                a = 1 / ((2 - diff_ss) / (dx @ V))
                v = np.clip(1 / a, 2, 10)
                if damping == 0:
                    damping = 1 / np.trace(M)
                    v /= 2
                damping *= v

            elif reduction_ratio > 0.75:
                damping /= 2

        x = new_x
        concs = new_concs
        prev_resid = resid
        sigma_hist.append(sigma)

        iterations += 1

        if iterations > max_iterations:
            if quiet_maxits:
                break

            ret = {
                "last_value": x,
                "jacobian": J,
                "weights": W,
                "residuals": resid,
                "concentrations": concs,
                "damping": damping,
                "sigma": sigma_hist,
                "iterations": iterations,
            }
            raise TooManyIterations(
                msg=("Maximum number of" "iterations reached"), last_value=ret
            )

    ret_extra = {
        "jacobian": J,
        "weights": W,
        "residuals": resid,
        "damping": damping,
        "sigma": sigma_hist,
        "iterations": iterations,
    }
    return x / 2.303, concs, ret_extra


def trivial_capping(x, dx):
    "Capping function where there is no capping"
    return x + dx


def max_ratio_capping(x, dx, ratio):
    "Capping to a fraction of change"
    aux = np.abs(dx) / x
    return np.where(aux > ratio, np.sign(dx) * (1 + ratio) * x, x + dx)


def fit_sigma(residuals, weights, npoints, nparams):
    return np.sqrt(weighted_sum_squares(residuals, weights) / (npoints - nparams))


def weighted_sum_squares(residuals, weights):
    return np.sum(weights * residuals**2)
