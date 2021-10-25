import math
import torch
import gpytorch
from matplotlib import pyplot as plt

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy


train_x = torch.linspace(0, 1, 10)
train_y = torch.sign(torch.cos(train_x * (4 * math.pi))).add(1).div(2)


class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class BetaBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    These are the only two functions that are needed for ELBO computation
    """
    def forward(self, function_samples, **kwargs):
        count, scale = kwargs["count"], kwargs["scale"]
        mus = torch.sigmoid(function_samples)
        alphas, betas = mus*scale, (1-mus)*scale
        return gpytorch.distributions.base_distributions.BetaBinomial(alphas, betas, count)
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

    
# Initialize model and likelihood
model = GPClassificationModel(train_x)
# likelihood = gpytorch.likelihoods.BernoulliLikelihood()
likelihood = BetaBinomialLikelihood()

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the number of training datapoints
mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

for i in range(training_iterations):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y, count=torch.ones(len(train_x)), scale=torch.ones(len(train_x)))
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
    
# Go into eval mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Test x are regularly spaced by 0.01 0,1 inclusive
    test_x = torch.linspace(0, 1, 101)
    z = model(test_x)
    # Get classification predictions
#     observed_pred = likelihood(model(test_x))
    print (z.variance)
    print (z.mean)
    # Initialize fig and axes for plot
#     f, ax = plt.subplots(1, 1, figsize=(4, 3))
#     ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
#     # Get the predicted labels (probabilites of belonging to the positive class)
#     # Transform these probabilities to be 0/1 labels
#     pred_labels = observed_pred.mean.ge(0.5).float()
#     ax.plot(test_x.numpy(), pred_labels.numpy(), 'b')
#     ax.set_ylim([-1, 2])
#     ax.legend(['Observed Data', 'Mean'])