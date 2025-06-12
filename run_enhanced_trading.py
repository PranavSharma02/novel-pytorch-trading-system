#!/usr/bin/env python3
"""
ULTRA-AGGRESSIVE Enhanced Trading System Runner
Optimized for MAXIMUM returns (with reduced risk constraints)
"""

import sys
import torch
from apply_return_boost import EnhancedTradingSystem

def main():
    stock_symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print("=" * 70)
    print("🔥 ULTRA-AGGRESSIVE MAXIMUM RETURN TRADING SYSTEM")
    print(f"   Target: MAXIMUM Annual Returns (Reduced Risk Constraints)")
    print(f"   📈 Trading Stock: {stock_symbol.upper()}")
    print("⚠️  WARNING: Ultra-aggressive mode prioritizes returns over risk control")
    print("=" * 70)
    
    # Initialize ULTRA-aggressive system
    enhanced_system = EnhancedTradingSystem(
        sequence_length=60,
        initial_balance=100000,
        return_boost_factor=1.25,  # 25% boost factor (increased)
        aggressive_mode=True
    )
    
    try:
        # 1. Load and prepare data
        print(f"\n📊 Phase 1: Data Loading for {stock_symbol}")
        result = enhanced_system.load_and_prepare_data(stock_symbol=stock_symbol)
        if result is None:
            return
        
        # 2. Create enhanced technical indicators
        print(f"\n🔧 Phase 2: Enhanced Technical Indicators Creation")
        enhanced_system.create_enhanced_technical_indicators()
        print(f"   Created {len(enhanced_system.features_df.columns)} enhanced features")
        
        # 3. Prepare sequences
        print(f"\n🔄 Phase 3: Sequence Preparation for Ultra-Aggressive Learning")
        enhanced_system.prepare_sequences()
        
        # 4. Train ULTRA-AGGRESSIVE ensemble
        print(f"\n🔥 Phase 4: ULTRA-AGGRESSIVE Maximum Return Training")
        enhanced_system.train_ultra_aggressive_ensemble(n_models=5, epochs=250, batch_size=96)
        
        # 5. Generate ULTRA-AGGRESSIVE signals
        print(f"\n🔥 Phase 5: ULTRA-AGGRESSIVE Signal Generation")
        signals = enhanced_system.generate_ultra_aggressive_signals()
        
        # 6. Backtest with ultra-aggressive signals
        print(f"\n💰 Phase 6: Ultra-Aggressive Performance Backtesting")
        results = enhanced_system.backtest_novel_signals(signals)
        
        # 7. Analyze ultra-aggressive performance
        print(f"\n📊 Phase 7: Ultra-Aggressive Performance Analysis")
        enhanced_system.analyze_performance(results)
        
        # Additional ultra-aggressive return-focused metrics
        if results:
            print(f"\n🔥 ULTRA-AGGRESSIVE RESULTS:")
            print(f"   Annual Return:       {results['annual_return']:.2%}")
            print(f"   Total Return:        {results['total_return']:.2%}")
            print(f"   Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown:        {results['max_drawdown']:.2%}")
            print(f"   Win Rate:            {results['win_rate']:.1%}")
            
            # Compare to buy & hold
            annual_alpha = results['annual_return'] - results['buy_hold_annual_return']
            print(f"   Annual Alpha:        {annual_alpha:.2%}")
            
            # Ultra-aggressive performance classification
            if results['annual_return'] > 0.25:  # 25%+ target
                print(f"   🔥 ULTRA SUCCESS: Maximum returns achieved!")
            elif results['annual_return'] > 0.20:  # 20%+ 
                print(f"   🚀 EXCELLENT: Very high returns!")
            elif results['annual_return'] > 0.15:  # 15%+
                print(f"   ✅ GOOD: High returns achieved")
            else:
                print(f"   ⚠️  REVIEW: Consider further ultra-aggressive optimization")
                
            # Risk-return ratio analysis
            return_to_risk = results['annual_return'] / max(abs(results['max_drawdown']), 0.01)
            print(f"   Return/Risk Ratio:   {return_to_risk:.2f}")
        
        print("\n" + "=" * 70)
        print(f"🔥 ULTRA-AGGRESSIVE TRADING SYSTEM COMPLETE FOR {stock_symbol.upper()}!")
        print("⚠️  Ultra-Aggressive Mode: Maximum Returns with Reduced Risk Constraints")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error in ultra-aggressive trading system: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 