import numpy as np

class GMM:
    def __init__(self, n_components, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.pi = np.full(self.n_components, 1 / self.n_components)
        self.mu = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.cov = np.array([np.eye(n_features)] * self.n_components)
        
        # Expectation-Maximization algorithm
        for _ in range(self.n_iter):
            # E-step: Calculate responsibilities
            responsibilities = self._expectation(X)
            
            # M-step: Update parameters
            self._maximization(X, responsibilities)
            
    def _expectation(self, X):
        n_samples, _ = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            gaussian = self._multivariate_gaussian(X, self.mu[k], self.cov[k])
            responsibilities[:, k] = self.pi[k] * gaussian
            
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities
    
    def _maximization(self, X, responsibilities):
        n_samples, n_features = X.shape
        total_responsibilities = np.sum(responsibilities, axis=0)
        epsilon = 1e-6  # Small positive value for regularization
        
        for k in range(self.n_components):
            # Update means
            self.mu[k] = np.dot(responsibilities[:, k], X) / total_responsibilities[k]
            
            # Update covariances
            diff = X - self.mu[k]
            cov = np.dot((responsibilities[:, k] * diff.T), diff) / total_responsibilities[k]
            cov += epsilon * np.eye(n_features)  # Regularization
            self.cov[k] = cov
            
            # Update mixing coefficients
            self.pi[k] = total_responsibilities[k] / n_samples
    
    def _multivariate_gaussian(self, X, mu, cov):
        n_features = X.shape[1]
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n_features * det)
        exp = np.exp(-0.5 * np.sum(np.dot((X - mu), inv) * (X - mu), axis=1))
        return norm_const * exp

    def predict(self, X):
        responsibilities = self._expectation(X)
        return np.argmax(responsibilities, axis=1)