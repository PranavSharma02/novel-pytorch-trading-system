"""
MAREA-Ensemble: Multi-Architecture Regime-Aware Ensemble for Adaptive Return Optimization

This module implements the core components of MAREA-Ensemble, a novel deep learning
trading system that combines:
1. Multi-architecture ensemble with diverse model configurations
2. Regime-aware market state detection and adaptive weighting
3. Ultra-aggressive return optimization with controlled risk management
4. Dynamic position sizing with neural network-based optimization

Research Paper: "MAREA-Ensemble: A Multi-Architecture Regime-Aware Deep Learning 
Framework for Ultra-Aggressive Stock Trading with Adaptive Risk Management"

Authors: [To be filled]
Institution: [To be filled]
Date: 2024

Key Innovation:
- Novel ensemble architecture combining 5 specialized models
- Regime-aware weighting system with 5 market states
- Ultra-aggressive loss function optimizing for maximum returns
- Multi-head attention mechanisms with positional encoding
- Dynamic position sizing with neural network adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MAREAUltraAggressiveLoss(nn.Module):
    """
    MAREA Ultra-Aggressive Loss Function
    
    Novel loss function designed for maximum return optimization while maintaining
    controlled risk exposure. Features multiple return-focused components and
    minimal risk penalties for aggressive trading strategies.
    
    Key Features:
    - Multi-component return optimization (Sharpe, magnitude, frequency, momentum)
    - Large return bonus mechanism for capturing significant market moves
    - Trend acceleration rewards for momentum trading
    - Minimal risk penalties allowing aggressive position taking
    
    Args:
        alpha (float): CVaR confidence level (default: 0.03 for ultra-aggressive)
        return_weight (float): Weight for return components vs risk penalties (default: 0.95)
        momentum_weight (float): Weight for momentum components (default: 0.05)
    """
    def __init__(self, alpha=0.03, return_weight=0.95, momentum_weight=0.05):
        super().__init__()
        self.alpha = alpha  # Reduced CVaR threshold for higher risk tolerance
        self.return_weight = return_weight
        self.momentum_weight = momentum_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        # Core return metrics
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        
        # 1. Traditional Sharpe ratio (reduced weight for ultra-aggressive mode)
        sharpe_ratio = mean_return / std_return
        
        # 2. Return magnitude component (high weight on absolute returns)
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        
        # 3. Positive return frequency (rewards frequent gains)
        positive_returns = (portfolio_returns > 0).float()
        positive_frequency = torch.mean(positive_returns)
        
        # 4. Large return bonus (NEW - rewards significant market captures)
        large_returns = torch.mean(torch.relu(portfolio_returns - 0.01))  # Bonus for returns > 1%
        
        # 5. Momentum capture with trend acceleration
        if len(portfolio_returns) > 1:
            momentum_consistency = torch.mean(
                torch.sign(portfolio_returns[1:]) * torch.sign(portfolio_returns[:-1])
            )
            # Trend acceleration bonus (NEW innovation)
            trend_acceleration = torch.mean(torch.relu(
                portfolio_returns[1:] - portfolio_returns[:-1]
            ))
        else:
            momentum_consistency = torch.tensor(0.0, device=signals.device)
            trend_acceleration = torch.tensor(0.0, device=signals.device)
            
        # 6. CVaR component (very relaxed for ultra-aggressive mode)
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        # 7. Signal turnover penalty (minimal for aggressive trading)
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        # Ultra-aggressive return optimization formula
        return_components = (
            0.25 * sharpe_ratio +              # Reduced Sharpe weight
            0.35 * return_magnitude +          # High absolute return focus
            0.20 * positive_frequency +        # Win rate optimization
            0.15 * large_returns +             # Big win capture bonus
            0.10 * momentum_consistency +      # Trend consistency
            0.08 * trend_acceleration          # Trend acceleration bonus
        )
        
        # Minimal risk penalties for ultra-aggressive mode
        risk_penalties = (
            0.02 * cvar +                      # Very minimal downside protection
            0.005 * signal_changes             # Almost no trading frequency penalty
        )
        
        total_loss = -(self.return_weight * return_components - risk_penalties)
        
        return total_loss

# Keep standard loss for backward compatibility
class MAREAStandardLoss(nn.Module):
    """Standard MAREA loss function for balanced trading"""
    def __init__(self, alpha=0.05, return_weight=0.9, momentum_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.return_weight = return_weight
        self.momentum_weight = momentum_weight
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        sharpe_ratio = mean_return / std_return
        return_magnitude = torch.mean(torch.abs(portfolio_returns))
        
        positive_returns = (portfolio_returns > 0).float()
        positive_frequency = torch.mean(positive_returns)
        
        if len(portfolio_returns) > 1:
            momentum_consistency = torch.mean(
                torch.sign(portfolio_returns[1:]) * torch.sign(portfolio_returns[:-1])
            )
        else:
            momentum_consistency = torch.tensor(0.0, device=signals.device)
            
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        return_components = (
            0.35 * sharpe_ratio +           
            0.30 * return_magnitude +       
            0.20 * positive_frequency +     
            0.15 * momentum_consistency     
        )
        
        risk_penalties = (
            0.05 * cvar +                   
            0.01 * signal_changes           
        )
        
        total_loss = -(self.return_weight * return_components - risk_penalties)
        return total_loss

class MAREAUltraAggressiveModel(nn.Module):
    """
    MAREA Ultra-Aggressive Model Architecture
    
    Advanced neural network model designed for maximum return capture with
    sophisticated attention mechanisms and multi-head prediction systems.
    
    Key Innovations:
    - 3-layer hierarchical GRU with enhanced capacity
    - 16-head multi-head attention for superior pattern recognition
    - Multi-head prediction system (return, momentum, volatility, trend)
    - Advanced feature extraction with progressive layer reduction
    - Ultra-aggressive signal combination with adaptive weighting
    
    Architecture:
    - Input → 3-Layer GRU → Multi-Head Attention → 2-Layer GRU → Feature Extraction → Multi-Head Output
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Hidden layer size (default: 192 for maximum capacity)
        dropout (float): Dropout rate (default: 0.12)
    """
    def __init__(self, input_size, hidden_size=192, dropout=0.12):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.model_name = "MAREA-Ultra-Aggressive"
        
        # Ultra-enhanced GRU with maximum capacity (3 layers)
        self.gru1 = nn.GRU(input_size, hidden_size, 3, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 2, dropout=dropout, batch_first=True)
        
        # Multi-head attention with enhanced capacity (16 heads)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=16, dropout=dropout, batch_first=True)
        
        # Advanced feature extraction with progressive reduction
        combined_feature_size = hidden_size//2 + hidden_size//2
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 512),  # High capacity entry
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),                    # Progressive reduction
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),                    # Continued reduction
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),                     # Final feature space
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # Multi-head prediction system (4 specialized heads)
        self.return_head = nn.Linear(64, 1)         # Primary return prediction
        self.momentum_head = nn.Linear(64, 1)       # Momentum signal prediction
        self.volatility_head = nn.Linear(64, 1)     # Volatility prediction
        self.trend_head = nn.Linear(64, 1)          # Trend strength prediction
        
    def forward(self, x):
        # Ultra-hierarchical GRU processing (3 layers)
        gru1_out, _ = self.gru1(x)
        
        # Enhanced multi-head attention mechanism (16 heads)
        attn_out, attn_weights = self.attention(gru1_out, gru1_out, gru1_out)
        
        # Secondary GRU processing (2 layers)
        gru2_out, hidden = self.gru2(attn_out)
        
        # Advanced global feature extraction
        final_hidden = hidden[-1]                   # Last hidden state
        global_avg = gru2_out.mean(dim=1)          # Global average pooling
        global_max = gru2_out.max(dim=1)[0]        # Global max pooling (unused in combination)
        
        # Feature combination
        combined_features = torch.cat([final_hidden, global_avg], dim=1)
        
        # Advanced feature extraction
        features = self.feature_extractor(combined_features)
        
        # Multi-head predictions
        return_pred = torch.tanh(self.return_head(features))
        momentum_pred = torch.tanh(self.momentum_head(features))
        volatility_pred = torch.sigmoid(self.volatility_head(features))
        trend_pred = torch.tanh(self.trend_head(features))
        
        # Ultra-aggressive signal combination with adaptive weighting
        volatility_weight = volatility_pred.squeeze()
        trend_weight = torch.abs(trend_pred.squeeze())
        
        # Advanced signal fusion favoring momentum and trend (ultra-aggressive formula)
        ultra_aggressive_signal = (
            0.4 * return_pred +                     # Primary return focus
            0.3 * momentum_pred +                   # High momentum weight
            0.2 * trend_pred +                      # Trend following
            0.1 * (volatility_weight.unsqueeze(1) * return_pred)  # Volatility-adjusted returns
        )
        
        return ultra_aggressive_signal

class MAREAStandardModel(nn.Module):
    """Standard MAREA model for balanced performance"""
    def __init__(self, input_size, hidden_size=160, dropout=0.15):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.model_name = "MAREA-Standard"
        
        # Enhanced GRU with larger capacity
        self.gru1 = nn.GRU(input_size, hidden_size, 2, dropout=dropout, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 1, dropout=dropout, batch_first=True)
        
        # Multi-head attention for trend capture
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout, batch_first=True)
        
        # Feature extraction layers
        combined_feature_size = hidden_size//2 + hidden_size//2
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multiple prediction heads
        self.return_head = nn.Linear(64, 1)
        self.momentum_head = nn.Linear(64, 1)
        self.volatility_head = nn.Linear(64, 1)
        
    def forward(self, x):
        gru1_out, _ = self.gru1(x)
        attn_out, _ = self.attention(gru1_out, gru1_out, gru1_out)
        gru2_out, hidden = self.gru2(attn_out)
        
        final_hidden = hidden[-1]
        global_features = gru2_out.mean(dim=1)
        
        combined_features = torch.cat([final_hidden, global_features], dim=1)
        features = self.feature_extractor(combined_features)
        
        return_pred = torch.tanh(self.return_head(features))
        momentum_pred = torch.tanh(self.momentum_head(features))
        volatility_pred = torch.sigmoid(self.volatility_head(features))
        
        volatility_weight = volatility_pred.squeeze()
        adaptive_signal = (
            (1 - volatility_weight).unsqueeze(1) * return_pred +
            volatility_weight.unsqueeze(1) * momentum_pred
        )
        
        return adaptive_signal

class MAREARegimeDetector(nn.Module):
    """
    MAREA Regime Detection System
    
    Advanced CNN-LSTM hybrid architecture for detecting market regimes and
    enabling regime-aware ensemble weighting.
    
    Detected Regimes:
    0: Bull Market - Strong upward trends
    1: Bear Market - Strong downward trends  
    2: Sideways Market - Low volatility, range-bound
    3: High Volatility - Elevated uncertainty
    4: Strong Momentum - Accelerating trends
    
    Architecture:
    - Input → 3-Layer CNN → Bidirectional LSTM → Regime Classification
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): LSTM hidden size (default: 80)
    """
    def __init__(self, input_size, hidden_size=80):
        super().__init__()
        
        self.regime_names = ["Bull", "Bear", "Sideways", "High-Vol", "Momentum"]
        
        # Enhanced CNN feature extraction (3 layers)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True, bidirectional=True)
        
        # Regime classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)  # 5 market regimes
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)
        conv_features = conv_features.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # Bidirectional LSTM temporal modeling
        _, (hidden, _) = self.lstm(conv_features)
        
        # Concatenate bidirectional hidden states
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Regime classification
        regime_probs = F.softmax(self.classifier(final_hidden), dim=1)
        
        return regime_probs

def get_marea_ultra_aggressive_configs():
    """
    MAREA Ultra-Aggressive Model Configurations
    
    Returns optimized configurations for each model in the ensemble,
    designed for maximum return capture with controlled risk.
    
    Returns:
        list: Configuration dictionaries for 5 specialized models
    """
    return [
        # Model 1: Ultra-Aggressive High-Capacity Model
        {'hidden_size': 192, 'dropout': 0.12, 'use_attention': True, 'ultra_aggressive': True, 
         'name': 'MAREA-Ultra-1'},
        
        # Model 2: High-Momentum Focus Model  
        {'hidden_size': 176, 'dropout': 0.14, 'use_attention': True, 'momentum_focus': True, 
         'aggressive': True, 'name': 'MAREA-Momentum'},
        
        # Model 3: Large Capacity Return-Optimized Model
        {'hidden_size': 160, 'dropout': 0.15, 'use_attention': True, 'return_focus': True, 
         'aggressive': True, 'name': 'MAREA-Return'},
        
        # Model 4: Enhanced Trend-Following Model
        {'hidden_size': 144, 'dropout': 0.16, 'use_attention': True, 'trend_focus': True, 
         'aggressive': True, 'name': 'MAREA-Trend'},
        
        # Model 5: High-Frequency Ultra-Aggressive Model
        {'hidden_size': 128, 'dropout': 0.18, 'use_attention': True, 'high_freq': True, 
         'ultra_aggressive': True, 'name': 'MAREA-HF'}
    ]

def get_marea_regime_weights():
    """
    MAREA Regime-Aware Weighting System
    
    Returns optimized weights for each model configuration based on 
    detected market regime, enabling adaptive ensemble behavior.
    
    Returns:
        dict: Regime-specific weight matrices for ensemble models
    """
    return {
        0: np.array([0.45, 0.25, 0.15, 0.10, 0.05]),  # Bull: Favor ultra-aggressive models
        1: np.array([0.15, 0.30, 0.25, 0.20, 0.10]),  # Bear: Balanced with momentum focus
        2: np.array([0.20, 0.25, 0.25, 0.20, 0.10]),  # Sideways: Momentum and return focus
        3: np.array([0.10, 0.20, 0.25, 0.30, 0.15]),  # High-Vol: Trend following emphasis
        4: np.array([0.50, 0.20, 0.15, 0.10, 0.05])   # Momentum: Maximum ultra-aggressive weight
    }

class MAREAPositionSizer(nn.Module):
    """
    MAREA Dynamic Position Sizing System
    
    Neural network-based position sizing that adapts to market conditions
    for optimal capital allocation and risk management.
    
    Args:
        input_size (int): Number of market feature inputs (default: 20)
    """
    def __init__(self, input_size=20):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, market_features):
        """Returns position size multiplier (0.5 to 1.5 range)"""
        base_size = self.network(market_features)
        return 0.5 + base_size  # Range: 0.5 to 1.5

# MAREA Training Configurations
MAREA_ULTRA_AGGRESSIVE_PARAMS = {
    'learning_rates': [0.002, 0.0018, 0.0015, 0.0012, 0.001],  # Progressive learning rates
    'batch_size': 96,                                           # Large batch size
    'epochs': 250,                                              # Extended training
    'patience': 25,                                             # Enhanced patience
    'weight_decay': 1e-6,                                       # Minimal regularization
    'gradient_clip': 2.0,                                       # Allow larger gradients
    'regime_detector_epochs': 80,                               # Regime detector training
    'position_sizer_epochs': 50,                                # Position sizer training
}

MAREA_STANDARD_PARAMS = {
    'learning_rates': [0.0015, 0.0012, 0.0010, 0.0008, 0.0006],
    'batch_size': 64,
    'epochs': 200,
    'patience': 20,
    'weight_decay': 5e-6,
    'gradient_clip': 1.5,
    'regime_detector_epochs': 60,
    'position_sizer_epochs': 30,
}

# Legacy aliases for backward compatibility
UltraReturnBoostLoss = MAREAUltraAggressiveLoss
ReturnBoostLoss = MAREAStandardLoss
UltraReturnOptimizedModel = MAREAUltraAggressiveModel
ReturnOptimizedModel = MAREAStandardModel
ReturnBoostRegimeDetector = MAREARegimeDetector
create_ultra_aggressive_model_configs = get_marea_ultra_aggressive_configs
get_ultra_aggressive_regime_weights = get_marea_regime_weights
PositionSizer = MAREAPositionSizer
ULTRA_AGGRESSIVE_TRAINING_PARAMS = MAREA_ULTRA_AGGRESSIVE_PARAMS
ENHANCED_TRAINING_PARAMS = MAREA_STANDARD_PARAMS 