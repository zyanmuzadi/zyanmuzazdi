"""
Boilerplate Physics-Informed Neural Network (PINN) for the 1D Burgers' equation.

Equation:
    u_t + u * u_x - nu * u_xx = 0

This script demonstrates how to:
1) Build a small fully-connected network with PyTorch.
2) Compute data loss (IC/BC supervision).
3) Compute the physics-informed loss using automatic differentiation.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SimplePINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, out_dim=1, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # Input is concatenated as [x, t]
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def burgers_physics_residual(model, x, t, nu):
    """Compute PDE residual f = u_t + u*u_x - nu*u_xx."""
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    # First derivatives
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivative with respect to x
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Burgers' equation residual
    f = u_t + u * u_x - nu * u_xx
    return f


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Viscosity coefficient
    nu = 0.01 / torch.pi

    # Create model + optimizer
    model = SimplePINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # ------------------------------------------------------------
    # Dummy training data placeholders (replace with real samples)
    # ------------------------------------------------------------
    n_ic = 128   # initial condition points
    n_bc = 128   # boundary condition points
    n_f = 1024   # collocation points for PDE residual

    # Domain: x in [-1, 1], t in [0, 1]
    x_ic = -1 + 2 * torch.rand(n_ic, 1, device=device)
    t_ic = torch.zeros(n_ic, 1, device=device)
    u_ic = -torch.sin(torch.pi * x_ic)  # common Burgers IC

    t_bc = torch.rand(n_bc, 1, device=device)
    x_bc_left = -torch.ones(n_bc // 2, 1, device=device)
    x_bc_right = torch.ones(n_bc - n_bc // 2, 1, device=device)
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
    u_bc = torch.zeros(n_bc, 1, device=device)  # e.g., homogeneous BC

    x_f = -1 + 2 * torch.rand(n_f, 1, device=device)
    t_f = torch.rand(n_f, 1, device=device)

    epochs = 2000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # --------------------------
        # Data loss: IC + BC fitting
        # --------------------------
        pred_ic = model(x_ic, t_ic)
        pred_bc = model(x_bc, t_bc)
        loss_ic = mse(pred_ic, u_ic)
        loss_bc = mse(pred_bc, u_bc)
        loss_data = loss_ic + loss_bc

        # -------------------------------------------------------------
        # Physics-Informed loss term:
        # Here we compute the Burgers PDE residual f(x,t)
        # and penalize it toward zero at collocation points.
        # -------------------------------------------------------------
        f = burgers_physics_residual(model, x_f, t_f, nu)
        loss_physics = mse(f, torch.zeros_like(f))

        # Total PINN loss
        loss = loss_data + loss_physics
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:4d} | total={loss.item():.3e} "
                f"data={loss_data.item():.3e} physics={loss_physics.item():.3e}"
            )

    print("Training finished.")


if __name__ == "__main__":
    main()
