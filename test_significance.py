#!/usr/bin/env python3
"""Quick test for significance_stars function."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from ggpubpy import significance_stars

def test_significance():
    """Test significance stars function."""
    test_cases = [
        (0.0001, "****"),  # 1e-4
        (0.001, "***"),   # 1e-3  
        (0.01, "**"),     # 1e-2
        (0.04, "*"),      # < 0.05
        (0.1, "ns"),      # >= 0.05
        (1e-5, "****"),   # < 1e-4
        (5e-4, "***"),    # > 1e-4 but < 1e-3
    ]
    
    print("Testing significance_stars function:")
    print("-" * 40)
    
    all_passed = True
    for p_val, expected in test_cases:
        result = significance_stars(p_val)
        status = "✓" if result == expected else "✗"
        print(f"{status} p={p_val:8.6f} -> {result:>4} (expected {expected})")
        if result != expected:
            all_passed = False
    
    print("-" * 40)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_significance()
