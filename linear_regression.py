import numpy as np

class LinearRegression():
    def __init__(self, X, y, alpha=0.03, n_iter=1500):

        # alpha adalah variabel yang digunakan menyimpan nilai alpha yang sudah ditentukan.
        # n_inters menunjukkan jumlah iterasi dari gradient descent.
        # n_samples menunjukkan jumlah sampel dari label.
        # n_features menunjukkan jumlah sampel dari fitur.
        # X menunjukkan nilai fitur yang sudah dinormalisasi.
        # y menunjukkan nilai label yang sudah dinormalisasi.
        # params menunjukkan nilai params yang sudah diinisialisasi dengan zero.
        # coef_ adalah variabel yang digunakan untuk menyimpan nilai koofesien hasil fitting.
        # intercept_ adalah variabel yang digunakan untuk menyimpan nilai intercept hasil fitting.

        self.alpha = alpha
        self.n_iter = n_iter
        self.n_samples = len(y)
        self.n_features = np.size(X, 1)
        self.X = np.hstack((np.ones(
            (self.n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.params = np.zeros((self.n_features + 1, 1))
        self.coef_ = None
        self.intercept_ = None

    def fit(self):
        # Fitting menggunakan gradient descent.
        # Parameter n_inters menunjukkan jumlah iterasi dari gradient descent.
        # Update rule ditentukan dengan script berikut: self.params - (self.alpha/self.n_samples) * \
        #             self.X.T @ (self.X @ self.params - self.y).

        for i in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * \
            self.X.T @ (self.X @ self.params - self.y)

        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

        return self

    def score(self, X=None, y=None):
        # Perhitungan R squared (coefficient of determination) pada variabel score.
        # Variabel y_pred memuat nilai hasil prediksi dari data X.

        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack((np.ones(
                (n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]

        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())
        return score

    def mean_absolute_error(self, X=None, y=None):
        # Perhitungan Mean Absolute Error pada variabel mae akan diterapkan sebagai cost function.
        # Variabel y_pred memuat nilai hasil prediksi dari data X.

        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack((np.ones(
                (n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]

        y_pred = X @ self.params
        d = y - y_pred
        mae = np.mean(abs(d))
        return mae

    def mean_squared_error(self, X=None, y=None):
        # Perhitungan Mean Squared Error pada variabel mse akan diterapkan sebagai cost function.
        # Variabel y_pred memuat nilai hasil prediksi dari data X.

        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack((np.ones(
                (n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]

        y_pred = X @ self.params
        d = y - y_pred
        mse = np.mean(d**2)
        return mse

    def predict(self, X):
        # Mendapatkan nilai label (y) berdasarkan fitur (X)

        n_samples = np.size(X, 0)
        y = np.hstack((np.ones((n_samples, 1)), (X-np.mean(X, 0)) \
                            / np.std(X, 0))) @ self.params
        return y