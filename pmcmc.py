import numpy as np
from scipy.stats import invgamma, norm
import pandas as pd
from Particle_Filter_project import Model,ModelData


import numpy as np
from scipy.stats import invgamma, norm
import pandas as pd

def compute_likelihood(observed_data, simulated_data, sigma_squared):
    residuals = observed_data - simulated_data
    likelihood = np.prod(norm.pdf(residuals, 0, np.sqrt(sigma_squared)))
    return likelihood

class PMMH:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.sigma_squared_current = 1.0
        self.sigma_squared_prior_alpha = 2
        self.sigma_squared_prior_beta = 2

    def propose_sigma_squared(self):
        return np.abs(np.random.normal(self.sigma_squared_current, 0.1))

    def compute_likelihood(self, sigma_squared):
        self.model.sigma_squared = sigma_squared
        simulated_data = self.model.run_simulation(T=len(self.data))
        likelihood = compute_likelihood(self.data, simulated_data, sigma_squared)
        return likelihood

    def compute_acceptance_ratio(self, sigma_squared_proposed, likelihood):
        prior_current = invgamma.pdf(self.sigma_squared_current, self.sigma_squared_prior_alpha, scale=self.sigma_squared_prior_beta)
        prior_proposed = invgamma.pdf(sigma_squared_proposed, self.sigma_squared_prior_alpha, scale=self.sigma_squared_prior_beta)
        acceptance_ratio = (likelihood * prior_proposed) / (likelihood * prior_current)
        return acceptance_ratio

    def run_pmcmc(self, iterations,store_iterations=None):
        posterior_samples = []
        stored_samples = {} 
        for _ in range(iterations):
            sigma_squared_proposed = self.propose_sigma_squared()
            likelihood = self.compute_likelihood(sigma_squared_proposed)
            acceptance_prob = self.compute_acceptance_ratio(sigma_squared_proposed, likelihood)
            if np.random.rand() < acceptance_prob:
                self.sigma_squared_current = sigma_squared_proposed
            posterior_samples.append(self.sigma_squared_current)
            if i in store_iterations:
                stored_samples[i] = posterior_samples.copy()
        return posterior_samples, stored_samples
    
 


def plot_prior_and_posteriors(prior_alpha, prior_beta, posterior_samples, stored_samples):
    x = np.linspace(0, max(posterior_samples) + 1, 500)
    prior_pdf = invgamma.pdf(x, prior_alpha, scale=prior_beta)

    plt.figure(figsize=(12, 6))
    plt.plot(x, prior_pdf, label="Prior", color="orange")

    for iteration, samples in stored_samples.items():
        sns.kdeplot(samples, label=f"Posterior at iteration {iteration}")

    plt.title("Evolution of Posterior Distribution in PMMH")
    plt.xlabel("sigma_squared")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

model = ModelData(K=1000, d=4, tau_m=1.0, tau_m_plus_1=2.0, A=0.01,
        sigma_squared=1.3, sigma_epsilon_squared=0.01, alpha=0.4,
        rho=np.random.rand(4, 4)) 
model.add_data('BMOQTreasuryQuotes.xlsx', usecols=range(12), index_col=0)
# Load your data - replace this with actual data loading
data = pd.read_excel('BMOQTreasuryQuotes.xlsx').to_numpy()
 # Example data, replace with actual data
store_iterations = [250, 500, 750]
# Run PMMH
pmcmc = PMMH(model, data)
posterior_samples,stored_samples = pmcmc.run_pmcmc(1000,store_iterations)
pmcmc = PMMH(model, data)
plot_prior_and_posteriors(prior_alpha=2, prior_beta=2, posterior_samples=posterior_samples, stored_samples=stored_samples)
