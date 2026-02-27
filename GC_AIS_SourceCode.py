"""
GC-AIS: Graph-Contrastive Autoencoder Instance Selection
Official Implementation

This script provides the core implementation of the GC-AIS framework,
including topological graph construction, GAT-based autoencoder, 
and the dual-branch (Generative + Contrastive) optimization strategy.
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# Reproducibility & CPU Optimization Settings
# ==========================================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(4) 

# ==========================================
# 1. Structure-Aware GAT Layer
# ==========================================
class DenseGATLayer(nn.Module):
    """
    Dense Graph Attention Layer customized for continuous topological graphs.
    """
    def __init__(self, in_features, out_features):
        super(DenseGATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a1 = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh = self.W(h)
        Wh1 = torch.matmul(Wh, self.a1)
        Wh2 = torch.matmul(Wh, self.a2)
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class GCAIS_Model(nn.Module):
    """
    Dual-branch Graph Autoencoder with GAT encoder and dense decoder.
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(GCAIS_Model, self).__init__()
        self.gat1 = DenseGATLayer(input_dim, hidden_dim)
        self.gat2 = DenseGATLayer(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, adj):
        h = self.gat1(x, adj)
        z = self.gat2(h, adj)
        x_hat = self.decoder(z)
        return z, x_hat

def construct_knn_adj(X, k=15, sigma=0.5):
    """
    Constructs a k-NN adjacency matrix using Cosine similarity and Gaussian Heat Kernel.
    """
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors(X)
    N = X.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            weight = np.exp(-(dist**2) / (sigma**2))
            adj[i, j] = weight
            adj[j, i] = weight
    return torch.tensor(adj, dtype=torch.float32)

# ==========================================
# 2. Automated Dataset Fetcher & Preprocessing
# ==========================================
def load_and_prep_dataset(name, max_samples=10000):
    """
    Fetches benchmark datasets from OpenML (equivalent to KEEL datasets).
    """
    print(f"\n[+] Downloading dataset '{name}'...")
    data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
    
    X_df = data.data.select_dtypes(include=[np.number])
    X_df = X_df.fillna(X_df.mean())
    X = X_df.values
    y = LabelEncoder().fit_transform(data.target)
    
    # Stratified sampling for memory efficiency on massive datasets
    if len(X) > max_samples:
        print(f"    * Sampling {max_samples} from {len(X)} instances for evaluation...")
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# ==========================================
# 3. Main GC-AIS Framework Runner
# ==========================================
def run_gc_ais(dataset_name, target_reduction=0.80, epochs=150):
    try:
        X, y = load_and_prep_dataset(dataset_name)
    except Exception as e:
        print(f"[-] Failed to load {dataset_name}. Error: {str(e)[:50]}")
        return

    N, d = X.shape
    print(f"--- Running GC-AIS on {dataset_name.upper()} (N={N}, d={d}) ---")
    start_time = time.time()
    
    # 1. Graph Construction Phase
    adj = construct_knn_adj(X, k=15)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # 2. Model Initialization
    model = GCAIS_Model(input_dim=d)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # 3. Dual-Branch Training Phase
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z, x_hat = model(X_tensor, adj)
        
        # Generative Branch (Reconstruction)
        loss_recon = F.mse_loss(x_hat, X_tensor)
        
        # Discriminative Branch (Structural Contrastive Loss - InfoNCE)
        pos_mask = (adj > 0).float()
        
        # Memory-efficient cosine similarity computation
        z_norm = F.normalize(z, p=2, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.t()) / 0.5 # Temperature = 0.5
        exp_sim = torch.exp(sim_matrix)
        
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)
        loss_cont = -torch.log(pos_sim / (all_sim + 1e-6) + 1e-6).mean()
        
        # Total Objective (lambda = 0.6)
        loss_total = 0.6 * loss_recon + 0.4 * loss_cont
        loss_total.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"    * Epoch {epoch+1}/{epochs} | Loss: {loss_total.item():.4f}")
        
    # 4. Pruning Phase (Dual Importance Scoring)
    model.eval()
    with torch.no_grad():
        z, x_hat = model(X_tensor, adj)
        
        # Calculate Reconstruction Confidence (RC)
        recon_errors = F.mse_loss(x_hat, X_tensor, reduction='none').mean(dim=1).numpy()
        RC = 1.0 / (recon_errors + 1e-6)
        
        # Calculate Structural Hardness (SH)
        z_norm = F.normalize(z, p=2, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.t())
        SH = 1.0 - (sim_matrix * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-6)
        SH = SH.numpy()
        
        # Normalize and Calculate Final Structural Importance Score (SIS)
        RC = MinMaxScaler().fit_transform(RC.reshape(-1, 1)).flatten()
        SH = MinMaxScaler().fit_transform(SH.reshape(-1, 1)).flatten()
        SIS = (1.5 * SH) - RC 
        
    num_retain = int(N * (1 - target_reduction))
    retained_indices = np.argsort(SIS)[-num_retain:]
    
    X_reduced = X[retained_indices]
    y_reduced = y[retained_indices]
    
    runtime = time.time() - start_time
    print(f"Reduction Target: {target_reduction*100:.0f}% -> Kept {len(X_reduced)} instances.")
    print(f"Execution Time: {runtime:.2f} seconds")
    
    # 5. Evaluation Phase
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    
    knn_orig = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn_orig.fit(X_train_orig, y_train_orig)
    acc_orig = accuracy_score(y_test, knn_orig.predict(X_test))
    
    knn_red = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn_red.fit(X_reduced, y_reduced)
    acc_red = accuracy_score(y_test, knn_red.predict(X_test))
    
    print(f"Accuracy (Original Data): {acc_orig*100:.2f}%")
    print(f"Accuracy (GC-AIS Reduced): {acc_red*100:.2f}%\n")

# ==========================================
# 4. Execution 
# ==========================================
datasets_to_run = [
    ('banana', 0.83),      
    ('phoneme', 0.85),     
    ('page-blocks', 0.77), 
    ('optdigits', 0.86),
    ('spambase', 0.89),    
    ('magic', 0.77),       
    ('texture', 0.86),     
    ('twonorm', 0.78)      
]

if __name__ == '__main__':
    for ds_name, reduction_rate in datasets_to_run:
        run_gc_ais(ds_name, target_reduction=reduction_rate, epochs=150)