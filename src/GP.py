import numpy as np
import torch
import gpytorch
import tqdm

import dataset

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, use_deep_kernel=True):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            if use_deep_kernel:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=5))
            else:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
            self.feature_extractor = DeepKernel(train_x.shape[-1])
            self.use_deep_kernel = use_deep_kernel

        def forward(self, x):
            if self.use_deep_kernel:
                projected_x = self.feature_extractor(x)
            else:
                projected_x = x
#             projected_x = projected_x - projected_x.min(0)[0]
#             projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
class GPHelper:
    def __init__(self, dataset: Dataset, ws_gp_x=None, ws_gp_y=None, explore_strategy='explore', use_deep_kernel=True):
        # Data for warm start to train likelihood etc.
        self.data = dataset
        self.explore_type = explore_strategy
        if ws_gp_x is None or ws_gp_y is None:
            ws_gp = self.data.warm_start_gp
        else:
            ws_gp = (ws_gp_x, ws_gp_y)
        
        self.use_deep_kernel = use_deep_kernel
        self.gp_model = self.gp_initialize(ws_gp[0], ws_gp[1])
        self.eval()
        
    def gp_initialize(self, train_gp_x, train_gp_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = GPRegressionModel(torch.from_numpy(np.array(train_gp_x)), torch.from_numpy(np.array(train_gp_y).astype(np.float32)), likelihood, self.use_deep_kernel)

        gp_model = gp_model.to(dev)
        likelihood = likelihood.to(dev)

        training_iterations = 100

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': gp_model.feature_extractor.parameters()},
            {'params': gp_model.covar_module.parameters()},
            {'params': gp_model.mean_module.parameters()},
            {'params': gp_model.likelihood.parameters()},
        ], lr=1e-3)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        def train():
            iterator = tqdm.tqdm(range(training_iterations))
            for i in iterator:
                # Zero backprop gradients
                optimizer.zero_grad()
                # Get output from model
                output = gp_model(torch.from_numpy(np.array(train_gp_x)).to(dev))
                # Calc loss and backprop derivatives
                loss = -mll(output, torch.from_numpy(np.array(train_gp_y).astype(np.float32)).to(dev))
                loss.backward()
                iterator.set_postfix(loss=loss.item())
                optimizer.step()

        %time train()
        print ("Noise: ", gp_model.likelihood.noise)
        return gp_model
        
    def eval(self):
        self.gp_model.eval()
        self.gp_model = self.gp_model.to(dev)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            z = self.gp_model(torch.from_numpy(self.data.arms).to(dev))
            zis = np.argsort(z.mean.cpu().numpy())

        ret = self.data.evaluate(mu_hat=z.mean.cpu().numpy())

        print (ret)
        return ret
    
    def explore(self, tol): 
        """
        tol: Number of examples allowed to explore
        """
        b, batch_size = 0, 5
        sample_per_arm = 6
        B = int(tol/(batch_size*sample_per_arm))

        while b < B:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                z = self.gp_model(torch.tensor(self.data.arms, dtype=torch.float).to(dev))
            var = z.variance.cpu().detach().numpy()

            if self.explore_type == 'exex':
                if b%2 == 0:
                    zis = np.argpartition(z.mean.cpu().detach().numpy() - z.variance.cpu().detach().numpy(), kth=batch_size)
                else:
                    zis = np.argpartition(-z.mean.cpu().detach().numpy() - z.variance.cpu().detach().numpy(), kth=batch_size)
            elif self.explore_type == 'explore':
                zis = np.argpartition(-z.variance.cpu().detach().numpy(), kth=batch_size)

            if b%10 == 0:
                print (z.mean.cpu().detach().numpy()[zis[:batch_size]])

            x, y = [], []
            for ni in range(batch_size):
                imgs, _ys = self.data.sample_arm(self.data.arms[zis[ni]], sample_per_arm)
                x += imgs
                y += _ys

            accs = self.data.eval_blackbox_model(x, y)
            new_accs = []
            for bi in range(batch_size):
                new_accs.append(np.mean(accs[bi*sample_per_arm: (bi+1)*sample_per_arm]))

            accs = new_accs
            new_gpp_x = torch.tensor(self.data.arms[zis[:batch_size]]).to(dev)
            new_gpp_y = torch.tensor(accs).to(dev)

            old_x, old_y = self.gp_model.train_inputs[0], self.gp_model.train_targets
            new_x, new_y = torch.cat([old_x, new_gpp_x], dim=0), torch.cat([old_y, new_gpp_y])
            # does not recompute the model params to maximiz likelihood
            self.gp_model.set_train_data(new_x, new_y, strict=False)
            # Not sure about the diff between get_fantasy_model and set_train_data. 
            # get_fantasy_model is recommended for small updates according to the docs.
            # Using this routine randomly threw CUDA_cholesky error that the matrix to be inversed is coming out to be singular.
            # Tried to understand the bug but could not. 
            # self.gp_model = self.gp_model.get_fantasy_model(new_gpp_x, new_gpp_y)

            b += 1
            
        new_x, new_y = self.gp_model.train_inputs[0], self.gp_model.train_targets
        self.gp_model = self.gp_initialize(new_x.detach().cpu().numpy(), new_y.detach().cpu().numpy())
        return self.gp_model