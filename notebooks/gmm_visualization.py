import numpy as np
import matplotlib.pyplot as plt; plt.style.use('/home/hugo/Thèse/styles.mplstyle')
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data
n_samples = 500
n_features = 2
n_classes = 3
random_state = 42

X, y_true = make_blobs(n_samples=n_samples, centers=n_classes, 
                       cluster_std=1.0, n_features=n_features, random_state=random_state)

# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=n_classes, covariance_type='full', random_state=random_state)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Function to draw ellipses
def plot_ellipse(position, covariance, ax, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    if covariance.shape == (2, 2):
        # Covariance matrix for 2D
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 5 * np.sqrt(eigenvalues)
    else:
        width, height = 2, 2
        angle = 0
    
    ellipse = Ellipse(position, width, height, angle=angle, alpha=0.2)
    ax.add_patch(ellipse)

# Create a meshgrid for visualization
x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500)
y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500)
Xx, Yy = np.meshgrid(x, y)
grid = np.c_[Xx.ravel(), Yy.ravel()]

# Plotting
fig, ax = plt.subplots(figsize=(16,9))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis', s=40, label='Data points')
ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=100, marker='x', label='Cluster Centers')

# Plot ellipses for each Gaussian component
for mean, cov in zip(gmm.means_, gmm.covariances_):
    plot_ellipse(mean, cov, ax, alpha=0.25, color='blue')

# Titles and labels
# plt.title('Gaussian Mixture Model with Ellipses')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
ax.set_axis_off()
plt.savefig('reports/figures/gmm_visualization.svg', bbox_inches='tight', dpi=300)
plt.show()
