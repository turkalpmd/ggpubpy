#!/usr/bin/env python3
"""
Pre-upload checklist script for PyPI publishing.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def check_files_exist() -> bool:
    """Check required files exist."""
    required_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "ggpubpy/__init__.py",
        "ggpubpy/plots.py",
        "ggpubpy/datasets.py",
    ]

    missing: List[str] = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    else:
        print("✅ All required files present")
        return True


def main() -> None:
    """Run pre-upload checklist."""
    print("🚀 PyPI Upload Pre-flight Checklist")
    print("=" * 40)

    checks: List[bool] = []

    # File existence check
    checks.append(check_files_exist())  # Package can be imported (with timeout)
    checks.append(
        run_command(
            'python -c "import ggpubpy; print(ggpubpy.__version__)"',
            "Package import test",
        )
    )

    # Run tests
    checks.append(run_command("python final_check.py", "Comprehensive tests"))

    # Clean and build (Windows compatible)
    checks.append(run_command("python -m build", "Package build"))

    # Check build artifacts
    checks.append(run_command("twine check dist/*", "Package validation"))
    # Summary
    print("\n" + "=" * 40)
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("✅ Ready for PyPI upload!")
        print("\nNext steps:")
        print("1. Upload to Test PyPI: twine upload --repository testpypi dist/*")
        print(
            "2. Test install: pip install --index-url https://test.pypi.org/simple/ ggpubpy"
        )
        print("3. Upload to PyPI: twine upload dist/* --username __token__")

        # Offer automated upload
        response = (
            input(
                "\nWould you like to automatically upload to Test PyPI first? (y/n): "
            )
            .lower()
            .strip()
        )
        if response == "y":
            print("\n🚀 Uploading to Test PyPI...")
            test_result = run_command(
                "twine upload --repository testpypi dist/*", "Test PyPI upload"
            )

            if test_result:
                print("✅ Test PyPI upload successful!")
                response2 = (
                    input("\nWould you like to upload to main PyPI now? (y/n): ")
                    .lower()
                    .strip()
                )
                if response2 == "y":
                    print("\n🚀 Uploading to main PyPI...")
                    pypi_result = run_command(
                        "twine upload dist/* --username __token__", "PyPI upload"
                    )
                    if pypi_result:
                        print("🎉 Package successfully published to PyPI!")
                        print("📦 Install with: pip install ggpubpy")
                        print("🌐 View at: https://pypi.org/project/ggpubpy/")
    else:
        print(f"❌ SOME CHECKS FAILED ({passed}/{total})")
        print("Please fix issues before uploading to PyPI")
        sys.exit(1)


if __name__ == "__main__":
    main()
