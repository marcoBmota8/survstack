import numpy as np

from . import functional as ssf

class SurvivalStacker:
    """Casts a survival analysis problem as a classification problem as
    proposed in Craig E., et al. 2021 (arXiv:2107.13480)
    """

    def __init__(self, times: np.ndarray | None = None, time_step: float | None = None,
                 time_encoding: str = 'onehot') -> None:
        """Generate a SurvivalStacker instance

        :param times: array of time points on which to create risk sets. If none, the
            times for all observed events are used.
        :param time_step: a base multiple on which to bin times. If none, the raw
            times are used.
        :param time_encoding: encoding to use for timepoints {"onehot","continuous"}
        """
        if time_encoding not in ssf.ALLOWED_ENCODINGS:
            raise ValueError(
                f"time_encoding must be one of {ssf.ALLOWED_ENCODINGS!r}, "
                f"got {time_encoding!r}"
            )

        self.time_step = time_step
        self.time_encoding = time_encoding
        self.times = times if times is not None else np.empty(0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Generate the risk time points

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        :return: self
        """
        event_field, time_field = y.dtype.names
        event_times = np.unique(y[time_field][y[event_field]])
        if self.time_step is None:
            self.times = event_times
        else:
            self.times = ssf.digitize_times(event_times, self.time_step)
        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) \
            -> tuple[np.ndarray, np.ndarray | None]:
        """Convert the input survival dataset to a stacked survival dataset

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field. If None, the returned dataset is constructed for
            evaluation.
        :return: a tuple containing the predictor matrix and response vector
        """
        if y is None:
            X_stacked = ssf.stack_eval(X, self.times, self.time_encoding)
            y_stacked = None
        else:
            X_stacked, y_stacked = ssf.stack_timepoints(X, y, self.times, self.time_encoding)
        return X_stacked, y_stacked

    def fit_transform(self, X: np.ndarray, y: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray | None]:
        """Fit to data, then transform it.

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        :return: a tuple containing the predictor matrix and response vector
        """
        self.fit(X, y)
        return self.transform(X, y)

    def predict_survival_function(self, estimates: np.ndarray) -> np.ndarray:
        """Calculate survival probability curves from hazard predictions.

        :param estimates: estimates as returned from a model trained on
        an evaluation set
        :return: array of survival probabilities for each sample over time.
        """
        return ssf.survival_function(estimates, self.times)


    def predict_cumulative_hazard_function(self, estimates: np.ndarray) -> np.ndarray:
        """Calculate cumulative hazard probability curves from hazard predictions.

        :param estimates: estimates as returned from a model trained on
        an evaluation set
        :return: array of survival probabilities for each sample over time.
        """
        return ssf.cumulative_hazard_function(estimates, self.times)

    def predict(self, estimates: np.ndarray) -> np.ndarray:
        """Calculate risk score from stacked survival estimates.

        :param estimates: estimates as returned from a model trained on
        an evaluation set
        :return: the risk score
        """
        return ssf.risk_score(estimates, self.times)

    def filter_prediction(self, y: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, ...]:
        """Filter predicted survival estimates to only include time points
        within the event range of the observed times. The resulting
        filtered data is suitable for use with scikit-survival's
        cumulative_dynamic_auc method.

        :param y: structured array of observed survival times
        :param prediction: survival or cumulative hazard predictions
        :return: a tuple of filtered predictions and times
        """
        event_field, time_field = y.dtype.names
        event_times = np.unique(y[time_field][y[event_field]])
        mask = (self.times >= event_times.min()) & (self.times <= event_times.max())
        return prediction[:, mask], self.times[mask]
