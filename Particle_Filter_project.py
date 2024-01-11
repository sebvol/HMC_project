

import numpy as np
from scipy.stats import norm, truncnorm, multivariate_normal
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import pandas as pd
from simulation_functions import simulate_sde, simulate_ou_process

class Model:
    def __init__(self, K, d, tau_m, tau_m_plus_1, A, sigma_squared, sigma_epsilon_squared, alpha, rho):
        '''
        K: Number of particles
        d: dimension
        tau_m: current time
        tau_m_plus_1: next step
        A：d × d matrices, controling mean convergence in d-dimensional Ornstein-Ulhenbeck process
        sigma_squared: variance
        sigma_epsilon_squared: noise variance
        alpha：proportional to psi or bid-ask spread
        rho：correlation coefficient
        x_m_k：
        y：mid-YtB D2D
        z：mid-YtB RFQ
        next_case：J_{m+1} the next case
        '''
        self.K = K
        self.d = d
        self.tau_m = tau_m
        self.tau_m_plus_1 = tau_m_plus_1
        self.A = A
        self.sigma_squared = sigma_squared
        self.sigma_epsilon_squared = sigma_epsilon_squared
        self.alpha = alpha
        self.rho = rho
        self.x_m_k = None
        self.y = None
        self.z = None
        self.next_case = None
        self.weights = np.zeros(self.K)


    def draw_half_bid_ask_spreads(self):
        '''
        Step 1 Drawing half bid-ask spreads
        '''
        mean = np.exp(-self.A * (self.tau_m_plus_1 - self.tau_m)) * np.prod(self.x_m_k, axis=1)
        Sigma = np.eye(self.d) * (self.tau_m_plus_1 - self.tau_m)
        x_m_hat_plus_1_k = np.array([np.random.multivariate_normal(mean_k * np.ones(self.d), Sigma) for mean_k in mean])
        psi_values = np.ones(self.d)
        psi_next = np.array([psi_values[i] * np.exp(x_m_hat_plus_1_k[:, i]) for i in range(self.d)]).T
        return psi_next
    


    def compute_weights(self):
        '''
        Step 2: Computing weights
        weights: weights fot particles
        '''
        for k in range(self.K):
            denominator = np.sqrt(self.sigma_squared * (self.tau_m_plus_1 - self.tau_m) + self.sigma_epsilon_squared)
            if self.next_case[k] == 1: 
                '''
                J1: D2C
                No truncation needed here
                '''
                self.weights[k] = norm.pdf(self.y[k], loc=self.alpha, scale=denominator)

            elif self.next_case[k] == 2: 
                '''
                J2: D2C
                No truncation needed here
                '''
                self.weights[k] = norm.pdf(self.y[k], loc=-self.alpha, scale=denominator)

            elif self.next_case[k] == 3: 
                '''
                J3: RFQ Buy 
                Right-sided truncation at zero
                '''
                a, b = -np.inf, (0 - self.alpha) / denominator
                self.weights[k] = truncnorm.cdf(self.y[k], a, b, loc=self.alpha, scale=denominator)

            elif self.next_case[k] == 4: 
                '''
                J4: RFQ Sell 
                Left-sided truncation at zero
                '''
                a, b = (0 - self.alpha) / denominator, np.inf
                self.weights[k] = truncnorm.cdf(self.y[k], a, b, loc=-self.alpha, scale=denominator)

            elif self.next_case[k] == 5: 
                '''
                J5: D2D 
                Two-sided truncation
                '''
                a = (self.z[k] - self.alpha) / denominator
                b = (self.z[k] + self.alpha) / denominator
                self.weights[k] = truncnorm.cdf(self.y[k], a, b, loc=self.z[k], scale=denominator)
        
        self.weights /= np.sum(self.weights)

        return self.weights

    def resampling(self, y_m_k, x_m_k, psi_m_k):
        '''
        Step 3 resampling
        '''
        indices = np.random.choice(self.K, size=self.K, p=self.weights)
        y_m_plus_1_k = y_m_k[indices]
        x_m_plus_1_k = x_m_k[indices]
        psi_next = psi_m_k[indices]
        return y_m_plus_1_k, x_m_plus_1_k, psi_next

    def draw_yj_m_hat_plus_1(self):
        '''
        Step 4
        '''
        yj_m_hat_plus_1 = np.zeros(self.K)
        
        for k in range(self.K):
            if self.next_case[k] == 1:
                yj_m_hat_plus_1[k] = self.y[k] + norm.rvs()  
            elif self.next_case[k] == 2:
                yj_m_hat_plus_1[k] = self.y[k] - norm.rvs()  
        return yj_m_hat_plus_1

    def draw_yj_m_plus_1(self, yj_m_hat_plus_1_k):
        '''
        Step 5
        '''
        variance = (self.sigma_squared * (self.tau_m_plus_1 - self.tau_m) * self.sigma_epsilon_squared) / \
                   (self.sigma_squared * (self.tau_m_plus_1 - self.tau_m) + self.sigma_epsilon_squared)
        yj_m_plus_1 = norm.rvs(loc=yj_m_hat_plus_1_k, scale=np.sqrt(variance), size=self.K)
        return yj_m_plus_1

    def draw_yj_m_hat_plus_1_k(self, y_m_plus_1, x_m_hat_plus_1):
        '''
        Step 6
        '''
        yj_m_hat_plus_1_k = np.zeros((self.K, self.d))
        Sigma = self.sigma_squared * (self.rho + self.rho.T) / 2  # Make Sigma symmetric
        np.fill_diagonal(Sigma, self.sigma_squared + self.sigma_epsilon_squared)
        
        # Check if Sigma is positive semidefinite
        eigenvalues, _ = eigh(Sigma)
        if np.any(eigenvalues < 0):
            # If not, add a small value to the diagonal until it becomes positive semidefinite
            min_eigenvalue = np.min(eigenvalues)
            Sigma += np.eye(self.d) * (-min_eigenvalue + 1e-8)

        for k in range(self.K):
            # Construct the mean vector mu_k for sample k
            mu_k = np.hstack([y_m_plus_1[k], x_m_hat_plus_1[k][:-1]])  # Ensure mu_k matches dimensions of Sigma
            
            # Draw a sample for each k
            yj_m_hat_plus_1_k[k] = multivariate_normal.rvs(mean=mu_k, cov=Sigma)

        return yj_m_hat_plus_1_k

    def run_simulation(self, T):

        ''' DATA '''        
        self.x_m_k = np.random.rand(self.K, self.d) 
        self.y = np.random.rand(self.K)  
        self.z = np.random.rand(self.K)  
        self.next_case = np.random.randint(1, 6, self.K)  

        results = [] 
        
        for m in range(T):
            ''' '''
            psi_next = self.draw_half_bid_ask_spreads() # Step 1
            self.weights = self.compute_weights() # Step 2
            self.y, self.x_m_k, self.psi_m_k = self.resampling(self.y, self.x_m_k, psi_next) # Step 3

            yj_m_hat_plus_1 = self.draw_yj_m_hat_plus_1() # Step 4
            yj_m_plus_1 = self.draw_yj_m_plus_1(yj_m_hat_plus_1) # Step 5
            yj_m_hat_plus_1_k = self.draw_yj_m_hat_plus_1_k(self.y, self.x_m_k) # Step 6
    
            results.append({
                "half_bid_ask_spreads": psi_next,
                "weights": self.weights,
                "resampled_values": (self.y, self.x_m_k, self.psi_m_k),
                "yj_m_hat_plus_1": yj_m_hat_plus_1,
                "yj_m_plus_1": yj_m_plus_1,
                "yj_m_hat_plus_1_k": yj_m_hat_plus_1_k,
            })
        
        return results
    
class ModelData(Model):
    
    
    def __init__(self, K, d, tau_m, tau_m_plus_1, A, sigma_squared, sigma_epsilon_squared, alpha, rho):
        super().__init__(K, d, tau_m, tau_m_plus_1, A, sigma_squared, sigma_epsilon_squared, alpha, rho)
    
            
    def add_data(self, file_name, **kwars):
        
        self.data = pd.read_excel(file_name, **kwars)
        
    def run_simulation(self, T):
        
        '''
        DATA
        '''
        
        self.x_m_k = simulate_ou_process(self.K, self.tau_m_plus_1 - self.tau_m, self.A, np.zeros(self.d), 0.1*np.eye(self.d), self.d, np.zeros(self.d))#np.random.rand(self.K, self.d) 
        self.y = simulate_sde(self.K, self.tau_m_plus_1 - self.tau_m, self.sigma_squared)
        self.z = np.random.rand(self.K)  
        # Draw from data
        self.next_case = np.array(self.data['j random'])
        
        results = [] 
        
        for m in range(T):
            ''' '''
            psi_next = self.draw_half_bid_ask_spreads() # Step 1
            self.weights = self.compute_weights(psi_next) # Step 2
            self.y, self.x_m_k, self.psi_m_k = self.resampling(self.y, self.x_m_k, psi_next) # Step 3

            yj_m_hat_plus_1 = self.draw_yj_m_hat_plus_1() # Step 4
            yj_m_plus_1 = self.draw_yj_m_plus_1(yj_m_hat_plus_1) # Step 5
            yj_m_hat_plus_1_k = self.draw_yj_m_hat_plus_1_k(self.y, self.x_m_k) # Step 6
    
            results.append({
                "half_bid_ask_spreads": psi_next,
                "weights": self.weights,
                "resampled_values": (self.y, self.x_m_k, self.psi_m_k),
                "yj_m_hat_plus_1": yj_m_hat_plus_1,
                "yj_m_plus_1": yj_m_plus_1,
                "yj_m_hat_plus_1_k": yj_m_hat_plus_1_k,
            })
        
        return results
        
        
    def compute_weights(self, psi_next):
        '''
        Step 2: Computing weights
        weights: weights fot particles
        '''
        
        for k in range(self.K):
            denominator = np.sqrt(self.sigma_squared * (self.tau_m_plus_1 - self.tau_m) + self.sigma_epsilon_squared)
            if self.next_case[k] == 1: 
                '''
                J1: D2C
                No truncation needed here
                '''
                self.weights[k] = norm.pdf(self.data['YtB'][k] + psi_next[k,0] - self.y[k], scale=denominator)

            elif self.next_case[k] == 2: 
                '''
                J2: D2C
                No truncation needed here
                '''
                
                self.weights[k] = norm.pdf(self.data['YtB'][k] - psi_next[k,0] - self.y[k], scale=denominator)

            elif self.next_case[k] == 3: 
                '''
                J3: RFQ Buy 
                Right-sided truncation at zero
                '''
                a, b = -np.inf, (0 - self.alpha) / denominator
                self.weights[k] = truncnorm.cdf(self.data['YtB'][k], a, b, loc=self.alpha, scale=denominator)

            elif self.next_case[k] == 4: 
                '''
                J4: RFQ Sell 
                Left-sided truncation at zero
                '''
                a, b = (0 - self.alpha) / denominator, np.inf
                self.weights[k] = truncnorm.cdf(self.data['YtB'][k], a, b, loc=-self.alpha, scale=denominator)

            elif self.next_case[k] == 5: 
                '''
                J5: D2D 
                Two-sided truncation
                '''
                a = (self.z[k] - self.alpha) / denominator
                b = (self.z[k] + self.alpha) / denominator
                self.weights[k] = truncnorm.cdf(self.y[k], a, b, loc=self.z[k], scale=denominator)
        
        self.weights /= np.sum(self.weights)

        return self.weights
    

    def draw_yj_m_hat_plus_1(self):
        '''
        Step 4
        '''
        yj_m_hat_plus_1 = np.zeros(self.K)
        
        for k in range(self.K):
            denominator = self.sigma_squared * (self.tau_m_plus_1 - self.tau_m) + self.sigma_epsilon_squared
            if self.next_case[k] == 1:
                yj_m_hat_plus_1[k] = self.data['YtB'][k] + norm.rvs()  
            elif self.next_case[k] == 2:
                yj_m_hat_plus_1[k] = self.data['YtB'][k] - norm.rvs()  
            elif self.next_case[k] == 3:
                yj_m_hat_plus_1[k] = truncnorm.rvs(a = self.z[k], b = np.inf, loc = self.data['YtB'][k], scale = denominator)
            elif self.next_case[k] == 4:
                yj_m_hat_plus_1[k] = truncnorm.rvs(a = -np.inf, b = self.z[k], loc = self.data['YtB'][k], scale = denominator)
            elif self.next_case[k] == 5:
                yj_m_hat_plus_1[k] = truncnorm.rvs(a = self.data['YtB'][k] - self.alpha, b = self.data['YtB'][k] - self.alpha,
                                                    loc = self.data['YtB'][k], scale = denominator)

        return yj_m_hat_plus_1

        


if __name__ == '__main__':
    
 
    
    model = ModelData(
        K=1000, d=4, tau_m=1.0, tau_m_plus_1=2.0, A=0.01,
        sigma_squared=1.3, sigma_epsilon_squared=0.01, alpha=0.4,
        rho=np.random.rand(4, 4)
    )
    
    model.add_data('BMOQTreasuryQuotes.xlsx', usecols=range(12), index_col=0)
    
    results = model.run_simulation(T=1000)

    y_hat = np.array([np.mean(result["yj_m_hat_plus_1_k"], axis=0) for result in results])
    plt.plot(y_hat[:, 0])
    plt.savefig('fig.png')
    
    trajectories = [run['half_bid_ask_spreads'][:, 0] for run in results]

    # Calculate the mean trajectory
    mean_trajectory = np.mean(np.vstack(trajectories), axis=0)
    
    plt.plot(mean_trajectory)
