from sklearn.decomposition import PCA
from sklearn import datasets
from QBCA import QBCA
if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    q = QBCA()
    q.fit(X)
    print(1)