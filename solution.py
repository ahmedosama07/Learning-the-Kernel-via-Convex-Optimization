import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class ConvexKernelLearning:
    """
    Implementation of kernel learning via convex optimization
    Based on the paper: "Learning the Kernel via Convex Optimization"
    """
    
    def __init__(self, base_kernels, regularization_param=0.1):
        """
        Initialize the kernel learning algorithm
        
        Args:
            base_kernels: List of base kernel functions
            regularization_param: Regularization parameter lambda
        """
        self.base_kernels = base_kernels
        self.lambda_reg = regularization_param
        self.kernel_weights = None
        self.alpha = None
        self.X_train = None
        self.y_train = None
        
    def gaussian_kernel(self, X1, X2, sigma):
        """Compute Gaussian kernel matrix"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
            
        # Compute pairwise squared distances
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True)
        distances = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
        
        return np.exp(-distances / (2 * sigma**2))
    
    def compute_base_kernels(self, X1, X2=None):
        """Compute all base kernel matrices"""
        if X2 is None:
            X2 = X1
            
        base_matrices = []
        for sigma in self.base_kernels:
            K = self.gaussian_kernel(X1, X2, sigma)
            base_matrices.append(K)
        return base_matrices
    
    def fit(self, X, y):
        """
        Learn optimal kernel weights using convex optimization
        
        Solves the problem:
        minimize: sum_i psi(y_i, z_i) + lambda * z^T G^dagger z
        subject to: G in convex hull of base kernels
        """
        self.X_train = X
        self.y_train = y
        m = len(X)  # number of training samples
        p = len(self.base_kernels)  # number of base kernels
        
        # Compute base kernel matrices
        base_matrices = self.compute_base_kernels(X)
        
        # Variables for the optimization problem
        theta = cp.Variable(p, nonneg=True)  # kernel weights
        z = cp.Variable(m)  # z = G*alpha (transformed variables)
        v = cp.Variable()   # bias term
        
        # Constraint: kernel weights sum to 1
        constraints = [cp.sum(theta) == 1]
        
        # Compute the combined kernel matrix G = sum(theta_i * K_i)
        G = sum(theta[i] * base_matrices[i] for i in range(p))
        
        # For the quadratic term z^T G^dagger z, we use the fact that
        # z^T G^dagger z <= t iff [t, z^T; z, G] >= 0 (Schur complement)
        t = cp.Variable()
        # Create Schur complement matrix: [[t, z^T], [z, G]]
        # Need to reshape z to be a column vector for proper matrix construction
        z_col = cp.reshape(z, (m, 1))
        z_row = cp.reshape(z, (1, m))
        
        # Build the Schur complement matrix block by block
        top_left = cp.reshape(t, (1, 1))
        top_right = z_row
        bottom_left = z_col
        bottom_right = G
        
        # Construct the full matrix
        top_block = cp.hstack([top_left, top_right])
        bottom_block = cp.hstack([bottom_left, bottom_right])
        schur_matrix = cp.vstack([top_block, bottom_block])
        
        constraints.append(schur_matrix >> 0)  # positive semidefinite
        
        # Objective function: quadratic loss + regularization
        predictions = z + v
        loss = cp.sum_squares(y - predictions) / m
        regularization = self.lambda_reg * t
        
        objective = cp.Minimize(loss + regularization)
        
        # Solve the convex optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            # Try ECOS solver first (more commonly available)
            problem.solve(solver=cp.ECOS, verbose=False)
            if problem.status not in ["infeasible", "unbounded"]:
                self.kernel_weights = theta.value
                self.z_optimal = z.value
                self.v_optimal = v.value
                
                # Compute alpha from z using pseudoinverse
                G_combined = sum(self.kernel_weights[i] * base_matrices[i] for i in range(p))
                self.alpha = np.linalg.pinv(G_combined) @ self.z_optimal
                
                print(f"Optimization successful with ECOS solver!")
                print(f"Optimal kernel weights: {self.kernel_weights}")
                print(f"Optimal objective value: {problem.value}")
                
                return self
            else:
                print(f"ECOS solver failed with status: {problem.status}")
                raise Exception("ECOS failed")
                
        except Exception as e:
            print(f"ECOS solver error: {e}")
            # Fallback to SCS solver
            try:
                problem.solve(solver=cp.SCS, verbose=False, max_iters=2000)
                if problem.status not in ["infeasible", "unbounded"]:
                    self.kernel_weights = theta.value
                    self.z_optimal = z.value
                    self.v_optimal = v.value
                    
                    G_combined = sum(self.kernel_weights[i] * base_matrices[i] for i in range(p))
                    self.alpha = np.linalg.pinv(G_combined) @ self.z_optimal
                    
                    print(f"Optimization successful with SCS solver!")
                    print(f"Optimal kernel weights: {self.kernel_weights}")
                    print(f"Optimal objective value: {problem.value}")
                    return self
                else:
                    print(f"SCS solver also failed: {problem.status}")
                    raise Exception("SCS failed")
            except Exception as e2:
                print(f"SCS solver error: {e2}")
                print("All solvers failed. Using uniform weights as fallback.")
                self.kernel_weights = np.ones(p) / p
                
                # For fallback, solve simple ridge regression
                base_matrices_np = [np.array(K) for K in base_matrices]
                G_uniform = sum(self.kernel_weights[i] * base_matrices_np[i] for i in range(p))
                
                # Add small regularization for numerical stability
                reg_matrix = G_uniform + self.lambda_reg * np.eye(m)
                self.alpha = np.linalg.solve(reg_matrix, y)
                self.v_optimal = 0
                
                print(f"Fallback: uniform kernel weights: {self.kernel_weights}")
                return self
    
    def predict(self, X_test):
        """Make predictions on test data"""
        if self.kernel_weights is None:
            raise ValueError("Model not fitted yet!")
        
        # Compute kernel matrix between test and training data
        base_matrices_test = self.compute_base_kernels(X_test, self.X_train)
        
        # Combine kernels with learned weights
        K_test = sum(self.kernel_weights[i] * base_matrices_test[i] for i in range(len(self.base_kernels)))
        
        # Make predictions: h(x) = sum_i alpha_i K(x_i, x) + v
        if hasattr(self, 'v_optimal') and self.v_optimal is not None:
            predictions = K_test @ self.alpha + self.v_optimal
        else:
            predictions = K_test @ self.alpha
        
        return predictions

def demonstrate_kernel_learning():
    """Demonstrate the kernel learning algorithm with synthetic data"""
    
    print("=== Kernel Learning via Convex Optimization Demo ===\n")
    
    # Generate synthetic regression data
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature dimension: {X_train.shape[1]}\n")
    
    # Define base kernels (Gaussian kernels with different bandwidths)
    # These correspond to different sigma values
    base_sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
    print(f"Base kernel bandwidths (sigma): {base_sigmas}")
    
    # Initialize and train the kernel learning algorithm
    kernel_learner = ConvexKernelLearning(
        base_kernels=base_sigmas,
        regularization_param=0.1
    )
    
    print("\n=== Training Kernel Learning Model ===")
    result = kernel_learner.fit(X_train, y_train)
    
    if result is not None:
        # Make predictions
        y_pred = kernel_learner.predict(X_test)
        
        # Compute performance metrics
        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        
        print(f"\n=== Results ===")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test Correlation: {correlation:.4f}")
        
        # Compare with baseline (uniform kernel weights)
        baseline_learner = ConvexKernelLearning(base_sigmas, 0.1)
        baseline_learner.kernel_weights = np.ones(len(base_sigmas)) / len(base_sigmas)
        baseline_learner.X_train = X_train
        baseline_learner.y_train = y_train
        
        # For baseline, we need to solve for alpha with uniform weights
        base_matrices = baseline_learner.compute_base_kernels(X_train)
        G_uniform = sum(baseline_learner.kernel_weights[i] * base_matrices[i] for i in range(len(base_sigmas)))
        
        # Solve simple ridge regression with uniform kernel
        try:
            alpha_uniform = np.linalg.solve(G_uniform + 0.1 * np.eye(len(X_train)), y_train)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudoinverse
            alpha_uniform = np.linalg.pinv(G_uniform + 0.1 * np.eye(len(X_train))) @ y_train
            
        baseline_learner.alpha = alpha_uniform
        baseline_learner.v_optimal = 0
        
        y_pred_baseline = baseline_learner.predict(X_test)
        rmse_baseline = np.sqrt(np.mean((y_test - y_pred_baseline)**2))
        correlation_baseline = np.corrcoef(y_test, y_pred_baseline)[0, 1]
        
        print(f"\n=== Comparison with Baseline (Uniform Weights) ===")
        print(f"Learned kernel RMSE: {rmse:.4f}")
        print(f"Baseline RMSE: {rmse_baseline:.4f}")
        print(f"Improvement: {(rmse_baseline - rmse)/rmse_baseline*100:.2f}%")
        
        print(f"\nLearned kernel correlation: {correlation:.4f}")
        print(f"Baseline correlation: {correlation_baseline:.4f}")
        
        # Visualize results
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Learned kernel weights
        plt.subplot(1, 3, 1)
        plt.bar(range(len(base_sigmas)), kernel_learner.kernel_weights, 
                color='skyblue', alpha=0.7)
        plt.xlabel('Base Kernel Index')
        plt.ylabel('Weight')
        plt.title('Learned Kernel Weights')
        plt.xticks(range(len(base_sigmas)), [f'σ={s}' for s in base_sigmas], rotation=45)
        
        # Plot 2: Predictions vs actual
        plt.subplot(1, 3, 2)
        plt.scatter(y_test, y_pred, alpha=0.7, label='Learned Kernel')
        plt.scatter(y_test, y_pred_baseline, alpha=0.7, label='Baseline')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual')
        plt.legend()
        
        # Plot 3: Feature space (if 2D)
        if X_train.shape[1] == 2:
            plt.subplot(1, 3, 3)
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7)
            plt.colorbar(label='Target Value')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Training Data in Feature Space')
        
        plt.tight_layout()
        plt.show()
        
        return kernel_learner
    else:
        print("Training failed!")
        return None

# Additional analysis functions
def analyze_kernel_properties(kernel_learner, X_train):
    """Analyze properties of the learned kernel"""
    if kernel_learner.kernel_weights is None:
        print("No trained model to analyze")
        return
    
    print("\n=== Kernel Analysis ===")
    
    # Compute combined kernel matrix
    base_matrices = kernel_learner.compute_base_kernels(X_train)
    G_combined = sum(kernel_learner.kernel_weights[i] * base_matrices[i] 
                    for i in range(len(kernel_learner.base_kernels)))
    
    # Analyze eigenvalues
    eigenvals = np.linalg.eigvals(G_combined)
    eigenvals = np.real(eigenvals[eigenvals > 1e-10])  # Remove numerical zeros
    
    print(f"Kernel matrix condition number: {np.max(eigenvals)/np.min(eigenvals):.2e}")
    print(f"Effective rank: {np.sum(eigenvals > 1e-6)}")
    print(f"Trace: {np.trace(G_combined):.4f}")
    
    # Plot eigenvalue spectrum
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(sorted(eigenvals, reverse=True), 'o-')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Spectrum of Learned Kernel')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.imshow(G_combined, cmap='viridis')
    plt.colorbar()
    plt.title('Learned Kernel Matrix')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the demonstration
    model = demonstrate_kernel_learning()
    
    if model is not None:
        # Additional analysis
        analyze_kernel_properties(model, model.X_train)
        
        print("\n=== Key Insights ===")
        print("1. The algorithm learns optimal weights for combining base kernels")
        print("2. This is formulated as a convex optimization problem")
        print("3. The Schur complement technique handles the pseudoinverse term")
        print("4. Performance typically improves over uniform kernel combination")
        print("5. The method is applicable to various kernel-based learning algorithms")