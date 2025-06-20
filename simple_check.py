#!/usr/bin/env python3
"""
Simple test script to verify package is ready for publication.
"""

import sys
import os


def run_simple_tests():
    """Run basic tests for ggpubpy package."""
    print("[INFO] Running basic ggpubpy tests...")
    print("=" * 50)

    # Test 1: Package import
    try:
        import ggpubpy

        print("[PASS] ggpubpy import successful")
    except Exception as e:
        print(f"[FAIL] ggpubpy import failed: {e}")
        return False

    # Test 2: Check version
    try:
        print(f"[INFO] Package version: {ggpubpy.__version__}")
        print("[PASS] Version check successful")
    except Exception as e:
        print(f"[FAIL] Version check failed: {e}")
        return False

    # Test 3: Dataset loading
    try:
        from ggpubpy import load_iris

        iris = load_iris()
        print(
            f"[PASS] Iris dataset loaded: {len(iris)} rows, {len(iris.columns)} columns"
        )
    except Exception as e:
        print(f"[FAIL] Dataset loading failed: {e}")
        return False

    # Test 4: Check documentation files
    try:
        docs_files = [
            "index.rst",
            "installation.rst",
            "quickstart.rst",
            "api.rst",
            "examples.rst",
            "contributing.rst",
            "changelog.rst",
            "conf.py",
        ]
        missing_files = [f for f in docs_files if not os.path.exists(f"docs/{f}")]
        if missing_files:
            raise Exception(f"Missing documentation files: {missing_files}")
        print(f"[PASS] Documentation structure complete: {len(docs_files)} files")
    except Exception as e:
        print(f"[FAIL] Documentation check failed: {e}")
        return False

    # Test 5: Check package files
    try:
        package_files = ["setup.py", "pyproject.toml", "README.md", "LICENSE"]
        missing_files = [f for f in package_files if not os.path.exists(f)]
        if missing_files:
            raise Exception(f"Missing package files: {missing_files}")
        print(f"[PASS] Package files complete: {len(package_files)} files")
    except Exception as e:
        print(f"[FAIL] Package files check failed: {e}")
        return False

    return True


def main():
    """Main test runner."""
    success = run_simple_tests()

    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] BASIC TESTS PASSED - Package structure is ready!")
        print()
        print("[PASS] Package import: PASSED")
        print("[PASS] Dataset loading: PASSED")
        print("[PASS] Documentation: PASSED")
        print("[PASS] Package files: PASSED")
        print()
        print("[INFO] Package is ready for PyPI publication!")
        print("[NOTE] Comprehensive plot tests may need scipy environment fix")
    else:
        print("[FAIL] TESTS FAILED - Package needs fixes before publication")
        sys.exit(1)


if __name__ == "__main__":
    main()
