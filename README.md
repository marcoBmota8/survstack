# survstack

**survstack** is a Python package implementing the survival stacking
method originally proposed in [*Survival stacking: casting survival
analysis as a classification problem* (2021)][ssref] and extended
in [*A review of survival stacking: a method to cast survival regression 
analysis as a classification problem* (2025)][ssref2] by Erin Craig,
Chenyang Zhong, and Robert Tibshirani.

**survstack** transforms time-to-event data into a stacked tabular 
format, replicating each sample’s covariates across relevant time points, 
enabling survival analysis standard classification algorithms. It supports 
discrete-time encoding (one-hot interval indicators) and continuous-time encoding 
(parametric time features), letting you choose the representation that 
best captures your hazard dynamics.

## Installation
```
pip install git+https://github.com/kaboroevich/survstack.git#egg=survstack
```

## SurvivalStacker Class

The recommended use is the provided SurvivalStacker class.
Survival data format follows that of the 
[scikit-survival][sksurv] package - a structured array with the first field indicating
the observation of an event as a boolean value, and the second
field denoting the survival time.

```python
import numpy as np
from survstack import SurvivalStacker
from sksurv.datasets import load_breast_cancer

X, y = load_breast_cancer()
X = X.loc[:, X.dtypes == np.float64].values

event_field, time_field = y.dtype.names
print(X.shape, y.shape, y[event_field].sum())
# (198, 78) (198,) 51

ss = SurvivalStacker(encoding='onehot')
X_stacked, y_stacked = ss.fit_transform(X, y)

print(X_stacked.shape, y_stacked.shape)
# (8117, 129) (8117,)
```

In the above example code, you can see the number of columns in X
increased by the number of observed events, while y became a
single column. The number of rows increases with respect to the
number of samples still under observation at each time-point.

[ssref]: https://doi.org/10.48550/arXiv.2107.13480
[ssref2]: https://doi.org/10.1515/ijb-2022-0055
[sksurv]: https://scikit-survival.readthedocs.io/en/stable/
