import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
import argparse
import os
import glob
warnings.filterwarnings('ignore')

from tradingPerformance import PerformanceEstimator

class MultiHeadAttention(nn.Module):
    """Enhanced Multi-head self-attention mechanism with positional encoding"""
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=60):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Positional encoding for better temporal awareness
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Self-attention
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(attn_output)
        return self.layer_norm(x + output)

class EnhancedGRUModel(nn.Module):
    """Enhanced GRU with Attention and Return-Focused Architecture"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, use_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        
        # Enhanced GRU with bidirectional capability
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         dropout=dropout, batch_first=True, bidirectional=False)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Second GRU for hierarchical learning
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 1, dropout=dropout, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size//2)
        
        # Enhanced dense layers for better return prediction
        self.fc1 = nn.Linear(hidden_size, 256)  # Increased capacity
        self.dropout1 = nn.Dropout(0.25)  # Reduced dropout for more capacity
        self.batch_norm3 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.15)
        self.batch_norm4 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.1)
        
        # Dual output heads for better signal generation
        self.return_head = nn.Linear(64, 1)  # Primary return prediction
        self.momentum_head = nn.Linear(64, 1)  # Momentum signal
        
    def forward(self, x):
        # First GRU layer
        gru_out, _ = self.gru(x)
        
        if self.use_attention:
            gru_out = self.attention(gru_out)
        
        # Second GRU layer
        gru_out2, hidden = self.gru2(gru_out)
        
        # Enhanced global features
        gru_final = hidden[-1]  # Last hidden state
        global_avg = gru_out2.mean(dim=1)  # Global average pooling
        global_max = gru_out2.max(dim=1)[0]  # Global max pooling for trend capture
        
        # Combine representations
        combined = torch.cat([gru_final, global_avg, global_max], dim=1)  # Enhanced features
        
        # Dense layers with enhanced capacity
        x = F.relu(self.batch_norm3(self.fc1(combined)))
        x = self.dropout1(x)
        
        x = F.relu(self.batch_norm4(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        # Dual outputs
        return_signal = torch.tanh(self.return_head(x))
        momentum_signal = torch.tanh(self.momentum_head(x))
        
        # Combine signals with learned weighting
        final_signal = 0.7 * return_signal + 0.3 * momentum_signal
        return final_signal

class ReturnOptimizedLoss(nn.Module):
    """Enhanced loss function optimized for higher returns with controlled risk"""
    def __init__(self, alpha=0.05, return_weight=0.85):
        super().__init__()
        self.alpha = alpha
        self.return_weight = return_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        # Enhanced return optimization
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        # Modified Sharpe ratio for higher return focus
        sharpe_ratio = mean_return / std_return
        
        # Return magnitude component (new)
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        
        # CVaR component (relaxed for higher returns)
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        # Signal consistency penalty (relaxed)
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        # Enhanced loss focusing on returns
        return_component = self.return_weight * (0.6 * sharpe_ratio + 0.4 * return_magnitude)
        risk_penalty = 0.08 * cvar + 0.02 * signal_changes  # Reduced penalties
        
        total_loss = -(return_component - risk_penalty)
        
        return total_loss

class RegimeDetector(nn.Module):
    """Enhanced market regime detection model"""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        # Enhanced CNN layers
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(32)
        
        # Enhanced LSTM
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(hidden_size * 2, 32)  # *2 for bidirectional
        self.fc2 = nn.Linear(32, 5)  # 5 regimes: Bull, Bear, Sideways, High-Vol, Momentum
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        
        x = x.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        _, (hidden, _) = self.lstm(x)
        # Concatenate final forward and backward hidden states
        x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        return F.softmax(self.fc2(x), dim=1)

class MetaLearner(nn.Module):
    """Enhanced meta-learner for adaptive ensemble with return focus"""
    def __init__(self, n_models=5, market_features=15):
        super().__init__()
        
        self.model_processor = nn.Linear(n_models, 32)  # Increased capacity
        self.market_processor = nn.Linear(market_features, 32)
        
        # Enhanced architecture
        self.fc1 = nn.Linear(64, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.15)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.1)
        
        # Multiple heads for different objectives
        self.prediction_head = nn.Linear(32, 1)
        self.confidence_head = nn.Linear(32, 1)
        self.volatility_head = nn.Linear(32, 1)  # For volatility prediction
        
    def forward(self, model_predictions, market_features):
        pred_processed = F.relu(self.model_processor(model_predictions))
        market_processed = F.relu(self.market_processor(market_features))
        
        combined = torch.cat([pred_processed, market_processed], dim=1)
        
        x = F.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        prediction = torch.tanh(self.prediction_head(x))
        confidence = torch.sigmoid(self.confidence_head(x))
        volatility = torch.sigmoid(self.volatility_head(x))
        
        return prediction, confidence, volatility

# Continue with the main PyTorchEnhancedTradingSystem class...
class PyTorchEnhancedTradingSystem:
    """
    Enhanced PyTorch Trading System optimized for higher returns
    
    Key improvements:
    - Return-focused loss function
    - Enhanced model architectures
    - Improved regime detection
    - Dual-head prediction models
    """
    
    def __init__(self, sequence_length=60, initial_balance=100000, device=None, return_focus=0.85):
        self.sequence_length = sequence_length
        self.initial_balance = initial_balance
        self.return_focus = return_focus
        
        # GPU Setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"🚀 Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"💎 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🔋 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.scaler = RobustScaler()
        self.models = []
        self.regime_detector = None
        self.meta_learner = None
        self.current_stock = None
        
    # [Include all the data loading and indicator creation methods from original...]
    # [This is a condensed version focusing on the key enhancement areas] 