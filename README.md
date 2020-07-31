# URL-MAI Quantiztion Based Clustering Algorithm
Quantization based clustering algorithm implementation done for the unsupervised learning subject using Numpy.

To run the paper experimentation execute main.py.

The algorithm works following standard sklearn function calls and is entirelly contained into the QBCA class.

Example:
```python
from sklearn import datasets
from qbca import QBCA

iris = datasets.load_iris()
X = iris.data
y = iris.target
q = QBCA()
X, y_test = q.fit_predict(X)

```
