import numpy as np


class SVM(object):
    '''
    Support Vector Machine implementation using Squential Minimal Optimaization Algorithm.

    '''
    def __init__(self, kernel='linear', C=1, max_iter=200, tol=0.001, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.b = None
        self.W = None
        self.cache = {'X': None, 'y': None}

    def linearKernel(self, xi, X) -> np.ndarray:
        ''' Linear kernel'''
        return xi @ X.T

    def gaussianKernel(self, xi, X, gamma) -> np.ndarray:
        ''' RBF kernel, a measure of similarity.'''
        m, n = X.shape

        if (gamma == 'auto') or (gamma == 'scale'):
            sigma = (1 / n) if gamma=='auto' else 1 / (n * X.var())
        elif type(gamma) == float:
            sigma = gamma
        else:
            raise ValueError(f"When 'gamma' is a string, it should be either 'scale' or 'auto'. Got '{gamma}' instead")

        g = lambda z, X: np.exp(-(np.linalg.norm(z - X, axis=1)**2) / (2 * sigma**2) )
        sim = np.apply_along_axis(lambda z: g(z, X.reshape(-1,n)), 1, xi.reshape(-1,n))

        return sim

    def compute_kernel(self, xi, X) -> np.ndarray:
        if self.kernel == 'linear':
            K = self.linearKernel(xi, X)
            return K
        elif self.kernel == 'rbf':
            K = self.gaussianKernel(xi, X, self.gamma)
            return K
        else:
            raise ValueError(f"Kernel of type '{self.kernel}' is unavailable.")

    def smo(self, X, y, C, tol) -> tuple:
        M, Nx = X.shape
        alpha = np.zeros(Nx)
        print(alpha.shape)
        b = 0
        passes = 0
        E = np.empty(M)
        K = self.compute_kernel(xi=X, X=X)
        while (passes < self.max_iter):
            # num_changed_alphas = 0.
            num_changed_alphas = 0
            fx = (alpha.reshape(-1,1) * y.reshape(-1,1)) * K + b
            # for i = 1, . . . m,
            for i in range(M):
                # Calculate Ei = f(x(i)) − y(i) using (2).
                E[i] = fx[i].squeeze() - y[i]
                # if ((y(i)Ei < −tol && αi < C) || (y(i)Ei > tol && αi > 0))
                if ( (y[i]*E[i] < -tol  and alpha[i] < C) or (y[i]*E[i] > tol and alpha[i] > 0) ):
                    # Select j ̸= i randomly.
                    j = np.random.choice(np.delete(np.arange(M), i), 1)
                    # Calculate Ej = f(x(j)) − y(j) using (2).
                    E[j] = fx[j] - y[j]
                    # Save old α’s
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                    # Compute L and H by (10) or (11).
                    if y[i] == y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(C, C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - C)
                        H = min(C, alpha[i] + alpha[j])
                    # if (L == H)
                    # continue to next i.
                    if L == H:
                        continue
                    # Compute η by (14).
                    nu = 2 * K[i,j] - K[i,i] - K[j,j]
                    # if (η >= 0)
                    # continue to next i.
                    if nu >= 0:
                        continue
                    # Compute and clip new value for αj using (12) and (15).
                    alpha[j] = alpha[j] - (y[j] * (E[i] - E[j]) / nu)
                    alpha[j] = np.clip(a=alpha[j], a_min=L, a_max=H)
                    # if (|αj − α(old)_j| < 10^−5)
                    # continue to next i.
                    if np.abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    # Determine value for αi using (16).
                    alpha[i] = alpha[i] + (y[i] * y[j] * (alpha_j_old - alpha[j]))
                    # Compute b1 and b2 using (17) and (18) respectively.
                    b1 = b - E[i] - (y[i] * (alpha[i] - alpha_i_old) * K[i,i]) - (y[j] * (alpha[j] - alpha_j_old) * K[i,j])
                    b2 = b - E[j] - (y[i] * (alpha[i] - alpha_i_old) * K[i,j]) - (y[j] * (alpha[j] - alpha_j_old) * K[j,j])
                    # Compute b by (19).
                    if (alpha[i] > 0) and (alpha[i] < C):
                        b = b1
                    elif (alpha[j] > 0) and (alpha[j] < C):
                        b = b2
                    else:
                        b = (b1 + b2)/2
                    # num changed alphas := num changed alphas + 1.
                    num_changed_alphas += 1
                # end if
            # end for
            # if (num changed alphas == 0)
            if num_changed_alphas == 0:
                # passes := passes + 1
                passes += 1
            # else
            else:
                # passes := 0
                passes = 0
        # end while
        return alpha, b

    def binary_classification(self, X, y):
        self.alpha, self.b = self.smo(X, y, self.C, self.tol)

    def fit(self, X, y):
        self.cache['X'] = X
        self.cache['y'] = y
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        if self.num_classes == 2:
            y[y == 0] = -1
            self.binary_classification(X, y)
        else:
            raise AttributeError('Number of classes cannot be more than 2.')

        self.fit_status_ = 1

    def decision_function(self, X):
        assert self.fit_status_, "NotFittedError: This SVM instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."

        K = self.compute_kernel(X, self.cache['X']).reshape(X.shape[0], -1)
        tmp = (self.alpha.reshape(-1,1) * self.cache['y'].reshape(-1,1)) * K.T
        margin = tmp.sum(axis=0)

        return margin

    def predict(self, X):
        assert self.fit_status_, "NotFittedError: This SVM instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."

        y_pred = self.decision_function(X)
        y_pred[y_pred >= 0] = 1

        return y_pred

