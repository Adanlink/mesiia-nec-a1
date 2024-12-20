import numpy as np

class NeuralNet:
    def __init__(self, layers, epochs, lr, momentum, activation, validation_split):
        """
        Initialize the neural network structure and hyperparameters.
        """
        np.random.seed(42)

        self.L = len(layers)                        # Number of layers
        self.n = layers                             # Units in each layer
        self.epochs = epochs                        # Number of epochs
        self.lr = lr                                # Learning rate
        self.momentum = momentum                    # Momentum
        self.fact = activation                      # Activation function
        self.validation_split = validation_split    # Fraction of data for validation

        # Initialize weights, thresholds, activations, and gradients
        self.w = [np.zeros((layers[l], layers[l - 1])) if l > 0 else None for l in range(self.L)]   # Weights
        self.theta = [np.zeros(layers[l]) if l > 0 else None for l in range(self.L)]                # Biases/thresholds
        self.xi = [np.zeros(layers[l]) for l in range(self.L)]                                      # Activations/outputs
        self.delta = [np.zeros(layers[l]) if l > 0 else None for l in range(self.L)]                # Error terms (deltas)
        self.d_w = [np.zeros((layers[l], layers[l - 1])) if l > 0 else None for l in range(self.L)] # Weight updates
        self.d_theta = [np.zeros(layers[l]) if l > 0 else None for l in range(self.L)]              # Bias updates
        self.d_w_prev = [np.zeros((layers[l], layers[l - 1])) if l > 0 else None for l in range(self.L)] # Prev weight updates (momentum)
        self.d_theta_prev = [np.zeros(layers[l]) if l > 0 else None for l in range(self.L)]         # Prev bias updates (momentum)
        self.h = [np.zeros(layers[l]) if l > 0 else None for l in range(self.L)]                    # Pre-activations

        self.loss_history = None

        # Activation function and derivative
        self.activation_map = {
            "sigmoid": (self.sigmoid, self.sigmoid_derivative),
            "relu": (self.relu, self.relu_derivative),
            "linear": (self.linear, self.linear_derivative),
            "tanh": (self.tanh, self.tanh_derivative),
        }
        self.g, self.g_prime = self.activation_map[activation]

    @staticmethod
    def sigmoid(h):
        return 1 / (1 + np.exp(-h))

    @staticmethod
    def sigmoid_derivative(h):
        g = 1 / (1 + np.exp(-h))
        return g * (1 - g)

    @staticmethod
    def relu(h):
        return np.maximum(0, h)

    @staticmethod
    def relu_derivative(h):
        return np.where(h > 0, 1, 0)

    @staticmethod
    def linear(h):
        return h

    @staticmethod
    def linear_derivative(h):
        return np.ones_like(h)

    @staticmethod
    def tanh(h):
        return np.tanh(h)

    @staticmethod
    def tanh_derivative(h):
        return 1 - np.tanh(h) ** 2

    def fit(self, X, y):
        """
        Train the network using the provided dataset.
        """
        # Split the dataset into training and validation sets
        n_samples = X.shape[0]
        validation_size = int(self.validation_split * n_samples)

        if validation_size > 0:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)      # shuffle again, so all new trainings have different orders
            X_train, X_val = X[indices[:-validation_size]], X[indices[-validation_size:]]
            y_train, y_val = y[indices[:-validation_size]], y[indices[-validation_size:]]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Randomly initialize weights and thresholds
        for l in range(1, self.L):
            self.w[l] = np.random.uniform(-0.5, 0.5, self.w[l].shape)
            self.theta[l] = np.random.uniform(-0.5, 0.5, self.theta[l].shape)

        self.loss_history = []

        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X_train[indices], y_train[indices]

            # Perform feed-forward and back-propagation on all training data
            self.feed_forward_batch(X_shuffled)
            self.back_propagate_batch(X_shuffled, y_shuffled)

            # Compute training loss
            train_loss = self.compute_loss_batch(X_train, y_train)
            val_loss = self.compute_loss_batch(X_val, y_val) if X_val is not None else None
            self.loss_history.append((train_loss, val_loss))

    def feed_forward_batch(self, X):
        """
        Perform feed-forward propagation for a batch of inputs.
        """
        self.xi[0] = X.T  # Input layer (transpose for matrix operations)
        for l in range(1, self.L):
            self.h[l] = self.w[l] @ self.xi[l - 1] - self.theta[l][:, np.newaxis]
            self.xi[l] = self.g(self.h[l])

    def back_propagate_batch(self, X, y):
        """
        Perform back-propagation to update weights and thresholds for a batch.
        """
        m = X.shape[0]  # Number of samples in the batch

        # Output layer errors
        delta_L = (self.xi[-1] - y.T) * self.g_prime(self.h[-1])
        self.delta[-1] = delta_L

        # Propagate errors backward
        for l in range(self.L - 2, 0, -1):
            self.delta[l] = (self.w[l + 1].T @ self.delta[l + 1]) * self.g_prime(self.h[l])

        # Update weights and thresholds
        for l in range(1, self.L):
            # Compute weight and bias updates
            d_w_l = -self.lr * (self.delta[l] @ self.xi[l - 1].T) / m + self.momentum * self.d_w_prev[l]
            d_theta_l = -self.lr * np.sum(self.delta[l], axis=1) / m + self.momentum * self.d_theta_prev[l]

            # Apply updates to weights and biases
            self.w[l] += d_w_l
            self.theta[l] += d_theta_l

            # Store updates for momentum
            self.d_w_prev[l], self.d_theta_prev[l] = d_w_l, d_theta_l

    def feed_forward(self, x):
        """
        Perform feed-forward propagation.
        """
        self.xi[0] = x  # Input layer
        for l in range(1, self.L):
            # Use efficient matrix operations for the linear combination
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            # Apply the activation function
            self.xi[l] = self.g(self.h[l])

    def back_propagate(self, x, y):
        """
        Perform back-propagation to update weights and thresholds.
        """
        # Compute output layer error (delta)
        self.delta[-1] = self.g_prime(self.h[-1]) * (self.xi[-1] - y)

        # Backpropagate errors through the hidden layers
        for l in range(self.L - 2, 0, -1):
            self.delta[l] = (self.w[l + 1].T @ self.delta[l + 1]) * self.g_prime(self.h[l])

        # Update weights and biases
        for l in range(1, self.L):
            # Compute weight and bias updates
            self.d_w[l] = -self.lr * np.outer(self.delta[l], self.xi[l - 1]) + self.momentum * self.d_w_prev[l]
            self.d_theta[l] = -self.lr * self.delta[l] + self.momentum * self.d_theta_prev[l]

            # Apply updates to weights and biases
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]

            # Store updates for momentum
            self.d_w_prev[l], self.d_theta_prev[l] = self.d_w[l], self.d_theta[l]

    def compute_loss_batch(self, X, y):
        """
        Compute quadratic error loss for a batch of inputs.
        """
        if X is None or y is None:
            return None

        # Perform feed-forward for all samples in the batch
        self.feed_forward_batch(X)

        # Compute mean squared error loss (vectorized)
        errors = self.xi[-1] - y.T
        total_loss = 0.5 * np.sum(errors**2) / X.shape[0]
        return total_loss

    def compute_loss(self, X, y):
        """
        Compute quadratic error loss.
        """
        total_loss = 0
        for x_sample, y_sample in zip(X, y):
            self.feed_forward(x_sample)
            total_loss += 0.5 * np.sum((self.xi[-1] - y_sample) ** 2)
        return total_loss / len(X)

    def predict(self, X):
        """
        Predict outputs for the given input.
        """
        predictions = []
        # for x in X:
        #     self.feed_forward_batch(x)
        #     predictions.append(self.xi[-1])
        self.feed_forward_batch(X)
        return self.xi[-1].T

    def loss_epochs(self):
        """
        Return training and validation loss history for all epochs.
        """
        return np.array(self.loss_history)
