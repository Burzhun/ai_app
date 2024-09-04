import sklearn


class BaseModel(sklearn.base.ClassifierMixin):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        return X, y

    def predict(self, X) -> list[float]:
        return [0.0] * len(X)
