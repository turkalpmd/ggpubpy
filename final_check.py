#!/usr/bin/env python3
"""
Final comprehensive test script to verify package is ready for publication.

This script tests:
- Package imports
- Core functionality 
- Dataset loading
- Plot generation
- API compatibility
"""

import sys
import os

def run_tests():
    """Run comprehensive tests for ggpubpy package."""
    print("🔍 Running comprehensive ggpubpy tests...")
    print("=" * 50)
    
    # Test 1: Package import
    try:
        import ggpubpy
        print("✅ ggpubpy import successful")
    except Exception as e:
        print(f"❌ ggpubpy import failed: {e}")
        return False
    
    # Test 2: Main functions import
    try:
        from ggpubpy import violinggplot, boxggplot, load_iris
        print("✅ Main functions import successful")
    except Exception as e:
        print(f"❌ Main functions import failed: {e}")
        return False
    
    # Test 3: Dataset loading
    try:
        iris = load_iris()
        print(f"✅ Iris dataset loaded: {len(iris)} rows, {len(iris.columns)} columns")
        print(f"   Species: {list(iris['species'].unique())}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test 4: Plot creation (non-interactive)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Test violin plot
        fig, ax = violinggplot(iris, x='species', y='sepal_length')
        plt.close()
        print("✅ Violin plot creation successful")
        
        # Test box plot
        fig, ax = boxggplot(iris, x='species', y='sepal_length')  
        plt.close()
        print("✅ Box plot creation successful")
        
        # Test with parameters
        fig, ax = violinggplot(iris, x='species', y='sepal_length', 
                              parametric=True, global_test=True, pairwise_test=True)
        plt.close()
        print("✅ Parametric violin plot with stats successful")
        
        fig, ax = boxggplot(iris, x='species', y='sepal_length',
                           parametric=False, global_test=True, pairwise_test=False)
        plt.close()
        print("✅ Non-parametric box plot with global test successful")
        
    except Exception as e:
        print(f"❌ Plot creation failed: {e}")
        return False
      # Test 5: Dataset utilities
    try:
        from ggpubpy.datasets import get_iris_palette, list_datasets
        palette = get_iris_palette()
        datasets = list_datasets()
        print(f"✅ Dataset utilities successful: {len(palette)} colors, {len(datasets)} datasets")
    except Exception as e:
        print(f"❌ Dataset utilities failed: {e}")
        return False
    
    # Test 6: Documentation structure
    try:
        import os
        docs_files = ['index.rst', 'installation.rst', 'quickstart.rst', 'api.rst', 
                     'examples.rst', 'contributing.rst', 'changelog.rst', 'conf.py']
        missing_files = [f for f in docs_files if not os.path.exists(f'docs/{f}')]
        if missing_files:
            raise Exception(f"Missing documentation files: {missing_files}")
        print(f"✅ Documentation structure complete: {len(docs_files)} files")
    except Exception as e:
        print(f"❌ Documentation check failed: {e}")
        return False
    
    return True

def main():
    """Main test runner."""
    success = run_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED - Package is ready for publication!")
        print()
        print("📦 Core functionality: ✅")
        print("📊 Plot generation: ✅") 
        print("🔧 API consistency: ✅")
        print("📈 Statistical tests: ✅")
        print("🎨 Color palettes: ✅")
        print("📊 Dataset loading: ✅")
        print("📚 Documentation: ✅")
        print()
        print("🚀 Ready to publish to PyPI!")
        print("🤝 Ready for community contributions!")
    else:
        print("❌ TESTS FAILED - Package needs fixes before publication")
        sys.exit(1)

if __name__ == "__main__":
    main()
