import torch
from torch.optim import AdamW
import logging

# Import CONFIG from main.py for global access
from main import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- SAM Stub ---
# This is a placeholder for the Sharpness-Aware Minimization (SAM) optimizer.
# A real SAM implementation would involve a two-step optimization process:
# 1. Compute gradients at the current point.
# 2. Perturb the weights in the direction of the gradients to find a "sharpness" point.
# 3. Compute gradients at this perturbed point.
# 4. Apply the optimizer step using the gradients from the perturbed point.
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        """
        Initializes the SAM optimizer (stub).
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer (torch.optim.Optimizer): The base optimizer (e.g., AdamW, SGD).
            rho (float): The neighborhood size parameter for SAM.
            adaptive (bool): Whether to use adaptive SAM (not implemented in stub).
        """
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        
        defaults = dict(rho=rho, adaptive=adaptive, enable_grad_scaling=False)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups # SAM uses base_optimizer's param groups
        self.defaults.update(self.base_optimizer.defaults) # Copy base optimizer defaults
        
        logger.info(f"SAM optimizer (stub) initialized with rho={rho}, adaptive={adaptive}")

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the first step of SAM: computes `e_w` and perturbs weights.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12) # Add epsilon for stability
            for p in group["params"]:
                if p.grad is None: continue
                # Store original weights
                self.state[p]["old_p"] = p.data.clone()
                # Perturb weights: w_hat = w + e_w
                e_w = p.grad * scale.to(p.grad.device)
                p.add_(e_w) # w_hat = w + e_w
        
        if zero_grad: self.zero_grad()
        logger.debug("SAM: First step (perturbation) completed.")

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Performs the second step of SAM: restores original weights and takes optimizer step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # Restore original weights
                p.data = self.state[p]["old_p"]
        
        self.base_optimizer.step() # Take the actual optimizer step
        if zero_grad: self.zero_grad()
        logger.debug("SAM: Second step (restoration and optimization) completed.")

    def _grad_norm(self):
        # Compute L2 norm of gradients
        shared_device = self.param_groups[0]["params"][0].device # Get a device for the norm calculation
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
                                         Required for SAM.
        """
        if closure is None:
            raise RuntimeError("SAM requires a closure for the second forward-backward pass.")
        
        # First forward-backward pass
        loss = closure()
        loss.backward() # Compute gradients at current point
        
        # First SAM step: perturb weights
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass at perturbed weights
        loss = closure()
        loss.backward() # Compute gradients at perturbed point
        
        # Second SAM step: restore weights and take optimizer step
        self.second_step(zero_grad=True)
        
        logger.debug("SAM: Full optimization step completed.")
        return loss

# --- make_optimizer function (Main function) ---
def make_optimizer(model_parameters):
    """
    Creates and returns an optimizer, optionally wrapped with SAM.
    Args:
        model_parameters (iterable): Iterable of model parameters to optimize.
    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    # Ensure CONFIG is accessible. If running standalone, provide a dummy.
    # In a real pipeline, main.CONFIG is globally available.
    current_config = CONFIG if 'CONFIG' in globals() else {
        'optimizer': {
            'lr': 1e-3,
            'wd': 1e-4,
            'sam': {
                'enabled': False,
                'rho': 0.05
            }
        }
    }

    base_opt = AdamW(model_parameters,
                     lr=current_config['optimizer']['lr'],
                     weight_decay=current_config['optimizer']['wd'])
    logger.info(f"Base optimizer (AdamW) initialized with lr={current_config['optimizer']['lr']}, wd={current_config['optimizer']['wd']}")

    if current_config['optimizer']['sam']['enabled']:
        optimizer = SAM(model_parameters, base_opt, rho=current_config['optimizer']['sam']['rho'])
        logger.info(f"SAM optimizer wrapper enabled with rho={current_config['optimizer']['sam']['rho']}")
        return optimizer
    else:
        logger.info("SAM optimizer wrapper disabled. Using base AdamW optimizer directly.")
        return base_opt

if __name__ == '__main__':
    # Example usage for testing the optimizer module
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Dummy model and data for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    dummy_input = torch.randn(1, 10)
    dummy_target = torch.randn(1, 1)

    # Dummy CONFIG for testing
    dummy_config_for_test = {
        'optimizer': {
            'lr': 1e-3,
            'wd': 1e-4,
            'sam': {
                'enabled': True, # Test with SAM enabled
                'rho': 0.05
            }
        }
    }
    # Temporarily set global CONFIG for testing if it's not already set
    if 'CONFIG' not in globals():
        global CONFIG
        CONFIG = dummy_config_for_test
    else: # If CONFIG exists, update it for the test
        CONFIG.update(dummy_config_for_test)

    # Test with SAM enabled
    print("\n--- Testing Optimizer with SAM Enabled ---")
    optimizer_sam = make_optimizer(model.parameters())
    criterion = nn.MSELoss()

    for i in range(5):
        def closure():
            optimizer_sam.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            return loss

        loss = optimizer_sam.step(closure)
        print(f"SAM Step {i+1}: Loss = {loss.item():.4f}")

    # Test with SAM disabled
    print("\n--- Testing Optimizer with SAM Disabled ---")
    CONFIG['optimizer']['sam']['enabled'] = False # Disable SAM for next test
    optimizer_no_sam = make_optimizer(model.parameters())

    for i in range(5):
        optimizer_no_sam.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer_no_sam.step()
        print(f"No SAM Step {i+1}: Loss = {loss.item():.4f}")
