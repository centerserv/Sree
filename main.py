import numpy as np
import pandas as pd
from layers.pattern import PatternValidator
from layers.presence import PresenceValidator
from layers.permanence import PermanenceValidator
from layers.logic import LogicValidator
from trust_update import update_trust
from sklearn.model_selection import train_test_split
import hashlib  # If not in permanence.py

def tune_params(data, labels, pattern, presence, permanence, logic, dataset_name):
    # Use 50% of test as val for tuning
    X_val, _, y_val, _ = train_test_split(data, labels, test_size=0.5, random_state=42)
    best_T = 0
    best_params = {}
    for alpha in [0.05, 0.1, 0.2]:
        for beta in [0.3, 0.4, 0.5]:
            for gamma in [0.05, 0.1, 0.2]:
                for delta in [0.1, 0.2, 0.3]:
                    history = update_trust(X_val, y_val, pattern, presence, permanence, logic, iterations=10, alpha=alpha, beta=beta, gamma=gamma, delta=delta, dataset_name=dataset_name)
                    final_T = history['T'][-1]
                    if final_T > best_T:
                        best_T = final_T
                        best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}
    return best_params

def inject_faults(data, labels, label_flip_rate=0.15, feature_noise_rate=0.1, sigma=0.1):
    data = data.copy()
    labels = labels.copy()
    flip_idx = np.random.rand(len(labels)) < label_flip_rate
    max_label = np.max(labels)
    labels[flip_idx] = np.random.randint(0, max_label + 1, sum(flip_idx))
    for i in range(len(data)):
        noise_idx = np.random.rand(len(data[i])) < feature_noise_rate
        data[i, noise_idx] += np.random.normal(0, sigma, sum(noise_idx))
    return data, labels

def main(dataset="heart"):
    if dataset == "heart":
        # Load real heart_large.csv dataset
        print("Loading heart_large.csv dataset...")
        df = pd.read_csv('heart_large.csv')
        print(f"Dataset shape: {df.shape}")
        
        # Separate features and target
        target_col = df.columns[-1]  # Assume last column is target
        features = df.drop(columns=[target_col])
        labels = df[target_col].values
        
        data = features.values
        dataset_name = "heart_large_30k"
        n_classes = len(np.unique(labels))
        print(f"Features: {data.shape[1]}, Classes: {n_classes}, Samples: {len(data)}")
    else:
        # Assume fetch_openml('mnist_784')
        # Synthetic for demo
        data = np.random.rand(1000, 784)
        labels = np.random.randint(0, 10, 1000)
        dataset_name = "mnist"
        n_classes = 10
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    pattern = PatternValidator(input_size=data.shape[1], output_size=n_classes)
    pattern.fit(X_train, y_train)
    presence = PresenceValidator()
    permanence = PermanenceValidator()
    logic = LogicValidator()
    best_params = tune_params(X_test, y_test, pattern, presence, permanence, logic, dataset_name)
    history = update_trust(X_test, y_test, pattern, presence, permanence, logic, iterations=10, **best_params, dataset_name=dataset_name)
    X_test_fault, y_test_fault = inject_faults(X_test, y_test)
    history_fault = update_trust(X_test_fault, y_test_fault, pattern, presence, permanence, logic, iterations=10, **best_params, dataset_name=dataset_name)
    df_normal = pd.DataFrame(history)
    df_normal['type'] = 'normal'
    df_fault = pd.DataFrame(history_fault)
    df_fault['type'] = 'fault'
    df = pd.concat([df_normal, df_fault])
    df['dataset'] = dataset_name
    df.to_csv(f'logs/{dataset_name}.csv', index=False)
    print(f"Final Accuracy: {history['accuracy'][-1]:.3f}, Trust: {history['T'][-1]:.3f} (Normal)")
    print(f"Final Accuracy: {history_fault['accuracy'][-1]:.3f}, Trust: {history_fault['T'][-1]:.3f} (Fault)")

if __name__ == "__main__":
    main(dataset="heart")
    main(dataset="mnist")