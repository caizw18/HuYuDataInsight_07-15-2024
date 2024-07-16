import numpy as np
from scipy.stats import norm

np.random.seed(0)

# True parameters of the two Gaussian distributions
mu_true = np.array([-2.0, 2.0])
sigma_true = np.array([1.0, 1.5])
weights_true = np.array([0.4, 0.6])

# Number of data points
num_samples = 1000

# Generate data points from the mixture distribution
component = np.random.choice(2, size=num_samples, p=weights_true)
data = np.random.normal(loc=mu_true[component], scale=sigma_true[component])

def gaussian(x, mu, sigma):
    return norm.pdf(x, mu, sigma)


def expectation(data, mu, sigma, weights):
    num_components = len(mu)
    num_samples = len(data)
    gamma = np.zeros((num_samples, num_components))

    # E-step: Calculate responsibilities
    for i in range(num_components):
        gamma[:, i] = weights[i] * gaussian(data, mu[i], sigma[i])

    gamma /= gamma.sum(axis=1, keepdims=True)

    return gamma


def maximization(data, gamma):
    num_samples, num_components = gamma.shape
    mu = np.zeros(num_components)
    sigma = np.zeros(num_components)
    weights = np.zeros(num_components)

    # M-step: Update parameters
    for i in range(num_components):
        weights[i] = gamma[:, i].mean()
        mu[i] = np.sum(gamma[:, i] * data) / np.sum(gamma[:, i])
        sigma[i] = np.sqrt(np.sum(gamma[:, i] * (data - mu[i]) ** 2) / np.sum(gamma[:, i]))

    return mu, sigma, weights


def em_algorithm(data, num_components, max_iter=100, tol=1e-4):
    num_samples = len(data)

    # Initialize parameters randomly
    mu = np.random.randn(num_components)
    sigma = np.random.rand(num_components)
    weights = np.ones(num_components) / num_components

    for _ in range(max_iter):
        # E-step
        gamma = expectation(data, mu, sigma, weights)

        # M-step
        mu_new, sigma_new, weights_new = maximization(data, gamma)

        # Check convergence
        if np.all(np.abs(mu_new - mu) < tol) and np.all(np.abs(sigma_new - sigma) < tol):
            break

        # Update parameters
        mu = mu_new
        sigma = sigma_new
        weights = weights_new

    return mu, sigma, weights

# Number of components in the Gaussian Mixture Model
num_components = 2

# Run EM algorithm
mu_estimated, sigma_estimated, weights_estimated = em_algorithm(data, num_components)

print("Estimated means:", mu_estimated)
print("Estimated standard deviations:", sigma_estimated)
print("Estimated weights:", weights_estimated)