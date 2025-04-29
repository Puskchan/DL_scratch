import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Dense_layer import Dense
from Activation import ReLU, TanH
from Loss import mse, mse_prime

# 1. Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # Convert categorical variables (like location) to numerical
    df = pd.get_dummies(df, columns=['location'])
    
    # Split features and target
    X = df.drop('price', axis=1).values
    y = df['price'].values.reshape(-1, 1)
    
    return X, y

# 2. Normalize the data
def normalize_data(X_train, X_test, y_train, y_test):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    
    return X_train, X_test, y_train, y_test, X_scaler, y_scaler

# 3. Create and train the neural network
def train_model(X_train, y_train, input_size):
    # Define network architecture
    network = [
        Dense(input_size, 64),  # Input layer
        ReLU(),
        Dense(64, 32),         # Hidden layer 1
        ReLU(),
        Dense(32, 16),         # Hidden layer 2
        ReLU(),
        Dense(16, 1)           # Output layer
    ]
    
    # Training parameters
    epochs = 1000
    learning_rate = 0.001
    batch_size = 32
    
    n_samples = len(X_train)
    
    for e in range(epochs):
        error = 0
        # Shuffle the data at each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Process each batch
        n_batches = (n_samples + batch_size - 1) // batch_size  # ceil division
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            output = batch_X
            for layer in network:
                output = layer.forward(output)
            
            # Calculate error
            error += mse(batch_y, output)
            
            # Backward pass
            grad = mse_prime(batch_y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        
        error /= n_batches
        if (e + 1) % 100 == 0:
            print(f'Epoch {e+1}/{epochs}, Error: {error}')
    
    return network

# 4. Main execution
def main():
    # Load your Pune housing dataset
    X, y = load_data('pune_housing_data.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize the data
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = normalize_data(
        X_train, X_test, y_train, y_test
    )
    
    # Train the model
    input_size = X_train.shape[1]
    network = train_model(X_train, y_train, input_size)
    
    # Make predictions
    def predict(X):
        output = X
        for layer in network:
            output = layer.forward(output)
        return output
    
    # Test the model
    test_predictions = predict(X_test)
    test_predictions = y_scaler.inverse_transform(test_predictions)
    actual_values = y_scaler.inverse_transform(y_test)
    
    # Calculate and print RMSE
    rmse = np.sqrt(np.mean((test_predictions - actual_values) ** 2))
    print(f'\nRoot Mean Square Error: {rmse:,.2f} rupees')

if __name__ == "__main__":
    main()