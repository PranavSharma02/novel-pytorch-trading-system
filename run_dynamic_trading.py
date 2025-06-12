#!/usr/bin/env python3
"""
Simplified Dynamic PyTorch Trading System Runner
Usage: python run_dynamic_trading.py [STOCK_SYMBOL]
Default: AAPL
"""

import sys
import subprocess
import os

def main():
    """Main function - run trading system with stock argument or default AAPL"""
    
    # Get stock symbol from argument or use default
    if len(sys.argv) > 1:
        stock_symbol = sys.argv[1].upper()
    else:
        stock_symbol = "AAPL"  # Default to Apple
    
    print(f"🚀 NOVEL PYTORCH TRADING SYSTEM")
    print(f"📈 Running analysis for: {stock_symbol}")
    print("=" * 60)
    
    # Run the trading system directly
    try:
        cmd = ["python", "pytorch_trading_system.py", "--stock", stock_symbol]
        
        print(f"🔄 Executing: {' '.join(cmd)}")
        print("=" * 60)
        
        # Run the command
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 60)
        print(f"✅ {stock_symbol} analysis completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running trading system for {stock_symbol}: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 