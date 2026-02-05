import torch
from torch.func import jacrev


def solve_fixed_point_with_bb(x_init, model_fn, input_size, max_iter: int=100, tol: float=1e-6, lr: float = 0.1, min_lr: float = 0.01, max_lr: float = 0.5, grad_clip: float = 1.0):
    """
    Solve fixed point equation x = f(x) using BB-based adaptive learning rate and Newton's method.
    NOTICE: IF GAIN(model(x).norm() / x.norm()) IS WAY LARGER THAN 1, THE METHOD MAY NOT CONVERGE AND BB ADAPTIVE LR MAY VERY LARGE AND UNSTABLE.
    If that happens, consider using Anderson acceleration instead or set a smaller grad_clip.
    
    Args:
        x_init: Initial guess for the fixed point
        model_fn: The function f(x) for which we want to solve x = f(x)
        input_size: Size of the input tensor
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        lr: Initial learning rate
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        grad_clip: Max update magnitude
    """
    x = x_init.clone().detach().requires_grad_(True)

    prev_x = None
    prev_g = None

    for i in range(max_iter):
        # 1. g(x) = f(x) - x
        fx = model_fn(x)
        g = fx - x
        res_norm = g.norm().item()
        
        # 2. BB Adaptive LR
        if prev_x is not None and prev_g is not None:
            with torch.no_grad():
                s = (x - prev_x).view(-1)  # Residual of movement
                y = (g - prev_g).view(-1)  # Residual of residuals
                
                # BB1 Formula: alpha = (s.T @ s) / |s.T @ y|
                dot_product = torch.dot(s, y)
                if dot_product.abs() > 1e-9:
                    lr_bb = (torch.dot(s, s) / dot_product.abs()).item()
                    # LR Clipping
                    lr = max(min_lr, min(lr_bb, max_lr))
                else:
                    lr = min_lr # Untrustable case fallback

        print(f"Iter {i}, Residual Norm: {res_norm:.4f}, Adaptive LR: {lr:.4f}")
        if res_norm < tol:
            print("Fixed point found!")
            break

        # 3. Jacobian
        J = jacrev(model_fn)(x).reshape(input_size, input_size)
        
        # 4. Newton step: (J - I) * delta_x = -g
        A = J - torch.eye(input_size, device=x.device)
        B = -g.view(-1)
        
        try:
            delta_x = torch.linalg.lstsq(A, B.float()).solution
        except Exception as e:
            print(f"Linear solver failed: {e}")
            break

        prev_x = x.clone().detach()
        prev_g = g.clone().detach()
        
        with torch.no_grad():
            delta_update = delta_x.view_as(x)
            # Gradient Clipping...
            if grad_clip > 0:
                norm = delta_update.norm()
                if norm > grad_clip:
                    delta_update = delta_update * (grad_clip / norm)
            x += lr * delta_update
            
    return x


def anderson_acceleration(x_init, model_fn, m=5, max_iter=100, tol=1e-6, lam=1e-4):
    """
    Solve fixed point equation x = f(x) using Anderson acceleration.
    Gain doesn't matter here since Anderson acceleration is adaptive.
    A little slower than BB-based method but very stable.

    Args:
        x_init: Initial guess for the fixed point
        model_fn: The function f(x) for which we want to solve x = f(x)
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        m: Anderson acceleration window size
        lam: Regularization parameter for least squares
    """
    x = x_init.clone().detach()
    
    # Store past m differences
    X = [] # x_{i+1} - x_i
    G = [] # g_{i+1} - g_i
    
    # Iter
    fx = model_fn(x)
    g = fx - x
    
    past_x = x.clone()
    past_g = g.clone()
    
    # Update
    x = fx.clone()

    for i in range(max_iter):
        fx = model_fn(x)
        g = fx - x
        res_norm = g.norm().item()
        
        print(f"Iter {i}, Residual Norm: {res_norm:.6f}")
        if res_norm < tol:
            break
            
        # Update differences
        current_x_diff = x - past_x
        current_g_diff = g - past_g
        
        X.append(current_x_diff.view(-1))
        G.append(current_g_diff.view(-1))
        
        if len(X) > m:
            X.pop(0)
            G.append(current_g_diff.view(-1))
            G.pop(0)
        
        # Solve a linear system: G * alpha = g
        matrix_G = torch.stack(G, dim=1) # (Size, m)
        matrix_X = torch.stack(X, dim=1) # (Size, m)
        
        # Solve for alpha
        try:
            # alpha: (G^T G + lam*I) alpha = G^T g
            GtG = matrix_G.T @ matrix_G
            Gtg = matrix_G.T @ g.view(-1, 1)
            alpha = torch.linalg.solve(GtG + lam * torch.eye(m, device=x.device), Gtg).view(-1)
            
            # Anderson Acceleration: x_{k+1} = f(x_k) - X @ alpha
            x = fx - (matrix_X @ alpha).view_as(x)
        except Exception:
            # Failed to solve, fallback to normal update
            x = fx.clone()

        past_x = x.clone()
        past_g = g.clone()

    return x
