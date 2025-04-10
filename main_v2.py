import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import random
import pickle  

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# Autoencoder Model Definition
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, bottleneck_size, dropout_rate=0.0):
        """
        The autoencoder consists of:
          - An encoder: two linear layers with ReLU activations and optional dropout.
          - A decoder: two linear layers with ReLU activations (except the last one is linear).
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_size, bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_size, input_size)
            # No activation here so that the output can take any real value.
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --------------------------------------
# Training and Evaluation Functions
# --------------------------------------
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    """Train the autoencoder for a fixed number of epochs."""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    return avg_loss

def get_reconstruction_errors(model, data_loader, criterion, device):
    """Compute the reconstruction MSE for each sample."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in data_loader:
            x_batch = batch[0].to(device)
            outputs = model(x_batch)
            batch_errors = torch.mean((x_batch - outputs) ** 2, dim=1)
            errors.extend(batch_errors.cpu().numpy())
    return np.array(errors)

# -------------------------------
# Genetic Algorithm Functions
# -------------------------------
param_ranges = {
    "hidden_size": [8, 16, 32],
    "bottleneck_size": [4, 8, 16],
    "learning_rate": [0.001, 0.005, 0.01, 0.05],
    "dropout_rate": [0.0, 0.2, 0.5]
}

def create_individual():
    """Create a new individual (a set of hyperparameters)."""
    return { key: random.choice(param_ranges[key]) for key in param_ranges }

def evaluate_individual(individual, X_val_tensor, input_size, device, batch_size=128, eval_epochs=20):
    """
    Evaluate an individual by:
      - Instantiating an autoencoder with the individual's hyperparameters.
      - Training for a fixed (short) number of epochs on the validation set.
      - Returning the average reconstruction error as the fitness.
    """
    dataset = TensorDataset(X_val_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Autoencoder(input_size, individual["hidden_size"],
                        individual["bottleneck_size"],
                        dropout_rate=individual["dropout_rate"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=individual["learning_rate"])
    train_model(model, loader, criterion, optimizer, eval_epochs, device)
    errors = get_reconstruction_errors(model, loader, criterion, device)
    avg_error = np.mean(errors)
    return avg_error

def crossover(parent1, parent2):
    """Combine two individuals by randomly selecting each hyperparameter from one of the parents."""
    return { key: random.choice([parent1[key], parent2[key]]) for key in parent1 }

def mutate(individual, mutation_rate=0.1):
    """With probability 'mutation_rate', randomly change each hyperparameter."""
    for key in individual:
        if random.random() < mutation_rate:
            individual[key] = random.choice(param_ranges[key])
    return individual

def run_ga(X_val_tensor, population_size, generations, input_size, device):
    """Run the genetic algorithm to optimize the autoencoder hyperparameters."""
    population = [create_individual() for _ in range(population_size)]
    best_losses = []
    best_individual = None
    best_loss = float('inf')
    
    for generation in range(generations):
        print(f"\n🔄 Generation {generation+1}")
        fitness_scores = []
        for individual in population:
            loss = evaluate_individual(individual, X_val_tensor, input_size, device)
            fitness_scores.append((loss, individual))
            print(f"{individual} ➤ Loss: {loss:.4f}")
        
        fitness_scores.sort(key=lambda x: x[0])
        if fitness_scores[0][0] < best_loss:
            best_loss = fitness_scores[0][0]
            best_individual = fitness_scores[0][1]
        best_losses.append(fitness_scores[0][0])
        
        top_individuals = [ind for (_, ind) in fitness_scores[:population_size // 2]]
        new_population = top_individuals.copy()
        while len(new_population) < population_size:
            parents = random.sample(top_individuals, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        population = new_population
        
    plt.plot(range(1, generations + 1), best_losses, marker='o')
    plt.title("GA Optimization Progress")
    plt.xlabel("Generation")
    plt.ylabel("Best Loss")
    plt.grid(True)
    plt.show()
    
    print(f"\n🏆 Best Hyperparameters: {best_individual} with Loss: {best_loss:.4f}")
    return best_individual

# -------------------------------
# Data Loading & Preprocessing
# -------------------------------
def load_credit_card_data(file_path):
    """
    Load and preprocess the Credit Card Fraud Detection dataset.
    Drops the 'Time' column, scales the features, and returns features, labels, and the fitted scaler.
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Time', 'Class']).values
    y = df['Class'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def load_nsl_kdd_data(train_path, test_path):
    """
    Load and preprocess the NSL-KDD dataset.
    Sets column names, converts label to binary (normal=0, attack=1),
    encodes categorical features, scales the features, and returns the scaler.
    """
    columns = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
        'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
        'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
        'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
        'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
        'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty_level'
    ]
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)
    df_train.columns = columns
    df_test.columns = columns
    
    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    combined_df = pd.concat([df_train, df_test], axis=0)
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        encoder = LabelEncoder()
        combined_df[col] = encoder.fit_transform(combined_df[col])
    df_train = combined_df.iloc[:len(df_train)]
    df_test = combined_df.iloc[len(df_train):]
    
    X_train = df_train.drop(columns=['label']).values
    y_train = df_train['label'].values
    X_test = df_test.drop(columns=['label']).values
    y_test = df_test['label'].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, scaler

# -------------------------------
# Experiment Functions & Model Saving
# -------------------------------
def run_credit_card_experiment():
    print("=== Credit Card Fraud Detection Experiment ===")
    
    file_path = "/creditcard.csv"  # Update the path if necessary
    X, y, scaler = load_credit_card_data(file_path)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train_full[y_train_full == 0]
    X_train_ga, X_val_ga = train_test_split(X_train, test_size=0.2, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[1]
    X_val_tensor = torch.tensor(X_val_ga, dtype=torch.float32)
    
    print("\nRunning GA Hyperparameter Optimization...")
    best_params = run_ga(X_val_tensor, population_size=6, generations=5, input_size=input_size, device=device)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = Autoencoder(input_size,
                        best_params["hidden_size"],
                        best_params["bottleneck_size"],
                        dropout_rate=best_params["dropout_rate"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    
    print("\nTraining final autoencoder...")
    final_epochs = 50
    train_model(model, train_loader, criterion, optimizer, final_epochs, device)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    errors = get_reconstruction_errors(model, test_loader, criterion, device)
    threshold = np.percentile(errors, 95)
    print(f"\nReconstruction error threshold (95th percentile): {threshold:.4f}")
    
    y_pred = (errors > threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nCredit Card Fraud Detection Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    normal_errors = errors[y_test == 0]
    fraud_errors = errors[y_test == 1]
    plt.figure(figsize=(10, 5))
    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal')
    plt.hist(fraud_errors, bins=50, alpha=0.6, label='Fraud')
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title("Reconstruction Error Distribution - Credit Card Dataset")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\nNumber of fraud samples in test set: {np.sum(y_test)}")
    
    # Save model and associated objects
    torch.save(model.state_dict(), "creditcard_model.pth")
    with open("creditcard_hyperparams.pkl", "wb") as f:
        pickle.dump(best_params, f)
    with open("creditcard_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Credit Card Model and associated objects saved successfully.")

def run_nsl_kdd_experiment():
    print("=== NSL-KDD Anomaly Detection Experiment ===")
    
    train_path = "/KDDTrain+.txt"  # Update path if necessary
    test_path = "/KDDTest+.txt"
    X_train, y_train, X_test, y_test, scaler = load_nsl_kdd_data(train_path, test_path)
    X_train_normal = X_train[y_train == 0]
    X_train_ga, X_val_ga = train_test_split(X_train_normal, test_size=0.2, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train_normal.shape[1]
    X_val_tensor = torch.tensor(X_val_ga, dtype=torch.float32)
    
    print("\nRunning GA Hyperparameter Optimization for NSL-KDD...")
    best_params = run_ga(X_val_tensor, population_size=6, generations=5, input_size=input_size, device=device)
    
    X_train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = Autoencoder(input_size,
                        best_params["hidden_size"],
                        best_params["bottleneck_size"],
                        dropout_rate=best_params["dropout_rate"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    
    print("\nTraining final autoencoder for NSL-KDD...")
    final_epochs = 50
    train_model(model, train_loader, criterion, optimizer, final_epochs, device)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    errors = get_reconstruction_errors(model, test_loader, criterion, device)
    threshold = np.percentile(errors, 95)
    print(f"\nReconstruction error threshold (95th percentile): {threshold:.4f}")
    
    y_pred = (errors > threshold).astype(int)
    print("\nNSL-KDD Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0))
    
    normal_errors = errors[y_test == 0]
    attack_errors = errors[y_test == 1]
    plt.figure(figsize=(10, 5))
    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal')
    plt.hist(attack_errors, bins=50, alpha=0.6, label='Attack')
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title("Reconstruction Error Distribution - NSL-KDD Dataset")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Save model and associated objects for NSL-KDD
    torch.save(model.state_dict(), "nslkdd_model.pth")
    with open("nslkdd_hyperparams.pkl", "wb") as f:
        pickle.dump(best_params, f)
    with open("nslkdd_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("NSL-KDD Model and associated objects saved successfully.")

if __name__ == "__main__":
    run_credit_card_experiment()
    run_nsl_kdd_experiment()
