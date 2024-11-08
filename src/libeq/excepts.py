class DivergedIonicStrengthWarning(UserWarning):
    """Warning for diverged ionic strength calculation."""

    def __init__(self, msg, last_value, **diagn):
        super().__init__()
        self.msg = msg
        self.last_value = last_value
        self.diagnostic = diagn

    def __str__(self):
        return self.msg


class FailedCalculateConcentrations(Exception):
    """When calculation of concentrations fails."""

    def __init__(self, msg, last_value, **diagn):
        super().__init__()
        self.msg = msg
        self.last_value = last_value
        self.diagnostic = diagn

    def __str__(self):
        return self.msg


class NotConvergenceException(Exception):
    """When convergence is not reached.

    It contains the last value of the iterations in case this information can
    be valuable.
    """

    def __init__(self, msg, last_value):
        super().__init__()
        self.msg = msg
        self.last_value = last_value

    def __str__(self):
        return self.msg


class TooManyIterations(NotConvergenceException):
    """When maximum number of iterations is reached.

    This exception is thrown when the maximum number of iterations
    has been reached without meeting the convergence criteria
    """
