import numpy as np
import pickle

def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required only for plotting utilities."
        ) from exc
    return plt

class Activation:
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1.0 - np.tanh(x)**2

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x):
        s = Activation.softmax(x)
        # Full Jacobian for each sample: diag(s) - s*s^T
        jacobian = -np.einsum('bi,bj->bij', s, s)
        idx = np.arange(s.shape[1])
        jacobian[:, idx, idx] += s
        return jacobian
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))

class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def bce(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def bce_derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.shape[0]

    @staticmethod
    def cce(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    @staticmethod
    def cce_derivative(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / y_pred) / y_true.shape[0]

class Layer:
    def __init__(self, input_size, output_size, activation='relu', weight_init='random_normal', **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init_name = weight_init
        self.init_params = {}
        
        # Init functions
        if activation == 'linear':
            self.activation = Activation.linear
            self.activation_derivative = Activation.linear_derivative
        elif activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        elif activation == 'softmax':
            self.activation = Activation.softmax
            self.activation_derivative = Activation.softmax_derivative
        elif activation == 'leaky_relu':
            alpha = kwargs.get('alpha', 0.01)
            self.activation = lambda x, a=alpha: Activation.leaky_relu(x, a)
            self.activation_derivative = lambda x, a=alpha: Activation.leaky_relu_derivative(x, a)
        elif activation == 'elu':
            alpha = kwargs.get('alpha', 1.0)
            self.activation = lambda x, a=alpha: Activation.elu(x, a)
            self.activation_derivative = lambda x, a=alpha: Activation.elu_derivative(x, a)
        else:
            raise ValueError("Unsupported activation function")
            
        seed = kwargs.get('seed', None)
        rng = np.random.default_rng(seed)
        if seed is not None:
            self.init_params['seed'] = seed
             
        # Initialize weights and biases with the same initialization strategy.
        if weight_init == 'zero':
            self.weights = np.zeros((input_size, output_size))
            self.biases = np.zeros((1, output_size))
        elif weight_init == 'random_uniform':
            lower = kwargs.get('lower_bound', -0.5)
            upper = kwargs.get('upper_bound', 0.5)
            self.weights = rng.uniform(lower, upper, (input_size, output_size))
            self.biases = rng.uniform(lower, upper, (1, output_size))
            self.init_params['lower_bound'] = lower
            self.init_params['upper_bound'] = upper
        elif weight_init == 'random_normal':
            mean = kwargs.get('mean', 0.0)
            variance = kwargs.get('variance', 0.1)
            if variance < 0:
                raise ValueError("variance must be non-negative")
            std = np.sqrt(variance)
            self.weights = rng.normal(mean, std, (input_size, output_size))
            self.biases = rng.normal(mean, std, (1, output_size))
            self.init_params['mean'] = mean
            self.init_params['variance'] = variance
        elif weight_init == 'xavier':
            x = np.sqrt(6.0 / (input_size + output_size))
            self.weights = rng.uniform(-x, x, (input_size, output_size))
            self.biases = np.zeros((1, output_size))
        elif weight_init == 'he':
            std = np.sqrt(2.0 / input_size)
            self.weights = rng.normal(0.0, std, (input_size, output_size))
            self.biases = np.zeros((1, output_size))
        else:
            raise ValueError("Unsupported weight init method")
         
        # Storing for backprop
        self.input = None
        self.z = None
        self.output = None
        self.d_weights = None
        self.d_biases = None
        
    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.biases
        self.output = self.activation(self.z)
        return self.output
        
    def backward(self, output_gradient, learning_rate, l1_lambda=0, l2_lambda=0, is_output_layer=False, loss_name=None):
        if is_output_layer and self.activation_name == 'softmax' and loss_name == 'categorical_crossentropy':
            d_z = output_gradient
        elif is_output_layer and self.activation_name == 'sigmoid' and loss_name == 'binary_crossentropy':
            d_z = output_gradient
        elif self.activation_name == 'softmax':
            # Jacobian-vector product without materializing per-sample Jacobians.
            s = self.output
            dot = np.sum(output_gradient * s, axis=1, keepdims=True)
            d_z = s * (output_gradient - dot)
        else:
            d_z = output_gradient * self.activation_derivative(self.z)
            
        input_gradient = np.dot(d_z, self.weights.T)
        self.d_weights = np.dot(self.input.T, d_z)
        self.d_biases = np.sum(d_z, axis=0, keepdims=True)
        
        if l1_lambda > 0:
            self.d_weights += l1_lambda * np.sign(self.weights)
        if l2_lambda > 0:
            self.d_weights += l2_lambda * self.weights
            
        # Weight update
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases
        
        return input_gradient

class FFNN:
    def __init__(self):
        self.layers = []
        self.loss_name = None
        self.loss_func = None
        self.loss_derivative = None
        self.l1_lambda = 0
        self.l2_lambda = 0
        
    def add(self, layer):
        self.layers.append(layer)
        
    def compile(self, loss='mse', l1_lambda=0, l2_lambda=0):
        self.loss_name = loss
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        if loss == 'mse': 
            self.loss_func = Loss.mse
            self.loss_derivative = Loss.mse_derivative
        elif loss == 'binary_crossentropy':
            self.loss_func = Loss.bce
            self.loss_derivative = Loss.bce_derivative
        elif loss == 'categorical_crossentropy':
            self.loss_func = Loss.cce
            self.loss_derivative = Loss.cce_derivative
        else:
            raise ValueError("Unsupported loss function")
            
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
        
    def backward(self, X, y_true, y_pred, learning_rate):
        if self.loss_name == 'categorical_crossentropy' and self.layers[-1].activation_name == 'softmax':
            # Combined derivative simplification
            output_gradient = (y_pred - y_true) / y_true.shape[0]
        elif self.loss_name == 'binary_crossentropy' and self.layers[-1].activation_name == 'sigmoid':
            output_gradient = (y_pred - y_true) / y_true.shape[0]
        else:
            output_gradient = self.loss_derivative(y_true, y_pred)
            
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            is_output_layer = (i == len(self.layers) - 1)
            output_gradient = layer.backward(
                output_gradient, 
                learning_rate, 
                self.l1_lambda, 
                self.l2_lambda,
                is_output_layer,
                self.loss_name
            )
            
    def fit(self, X_train, y_train, epochs, batch_size, learning_rate, validation_data=None, verbose=1):
        history = {'loss': [], 'val_loss': []}
        import sys
        
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data for stochasticity
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            # Batch processing
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward Pass
                y_pred_batch = self.forward(X_batch)
                
                # Calculate Loss
                loss = self.loss_func(y_batch, y_pred_batch)
                
                # Add Regularization Loss
                l1_penalty = 0
                l2_penalty = 0
                for layer in self.layers:
                    if self.l1_lambda > 0:
                        l1_penalty += self.l1_lambda * np.sum(np.abs(layer.weights))
                    if self.l2_lambda > 0:
                        l2_penalty += 0.5 * self.l2_lambda * np.sum(layer.weights ** 2)
                
                loss += l1_penalty + l2_penalty
                epoch_loss += loss * len(X_batch)
                
                # Backward Pass
                self.backward(X_batch, y_batch, y_pred_batch, learning_rate)
                
                if verbose == 1:
                    progress = int(20 * min(i + batch_size, num_samples) / num_samples)
                    bar = '=' * progress + '.' * (20 - progress)
                    sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} [{bar}]")
                    sys.stdout.flush()
                
            epoch_loss /= num_samples
            history['loss'].append(epoch_loss)
            
            # Validation
            val_loss = None
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_func(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
                
            if verbose == 1:
                val_msg = f"- val_loss: {val_loss:.4f}" if validation_data else ""
                print(f" - loss: {epoch_loss:.4f} {val_msg}")
                
        return history
        
    def predict(self, X):
        return self.forward(X)
        
    def plot_weight_distribution(self, layer_indices=None):
        plt = _get_plt()
        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))
            
        num_layers = len(layer_indices)
        fig, axes = plt.subplots(num_layers, 1, figsize=(6, 4 * num_layers))
        if num_layers == 1:
            axes = [axes]
            
        for i, idx in enumerate(layer_indices):
            layer = self.layers[idx]
            axes[i].hist(layer.weights.flatten(), bins=50, alpha=0.7)
            axes[i].set_title(f'Layer {idx+1} Weight Distribution')
            
        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layer_indices=None):
        plt = _get_plt()
        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))
            
        num_layers = len(layer_indices)
        fig, axes = plt.subplots(num_layers, 1, figsize=(6, 4 * num_layers))
        if num_layers == 1:
            axes = [axes]
            
        for i, idx in enumerate(layer_indices):
            layer = self.layers[idx]
            if layer.d_weights is not None:
                axes[i].hist(layer.d_weights.flatten(), bins=50, alpha=0.7, color='orange')
                axes[i].set_title(f'Layer {idx+1} Gradient Distribution')
            else:
                axes[i].set_title(f'Layer {idx+1} Gradient (None)')
                
        plt.tight_layout()
        plt.show()
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            model_state = {
                'layers': self.layers,
                'loss_name': self.loss_name,
                'l1_lambda': self.l1_lambda,
                'l2_lambda': self.l2_lambda,
            }
            pickle.dump(model_state, f)
             
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            saved_obj = pickle.load(f)

        # Backward compatibility with older checkpoints that only saved layers.
        if isinstance(saved_obj, list):
            self.layers = saved_obj
            self.loss_name = None
            self.loss_func = None
            self.loss_derivative = None
            self.l1_lambda = 0
            self.l2_lambda = 0
            return

        if not isinstance(saved_obj, dict) or 'layers' not in saved_obj:
            raise ValueError("Unsupported checkpoint format")

        self.layers = saved_obj['layers']
        self.loss_name = saved_obj.get('loss_name')
        self.l1_lambda = saved_obj.get('l1_lambda', 0)
        self.l2_lambda = saved_obj.get('l2_lambda', 0)

        if self.loss_name is None:
            self.loss_func = None
            self.loss_derivative = None
        else:
            self.compile(
                loss=self.loss_name,
                l1_lambda=self.l1_lambda,
                l2_lambda=self.l2_lambda,
            )