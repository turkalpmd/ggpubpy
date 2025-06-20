@echo off
REM Windows batch script for ggpubpy development commands
REM Usage: build.bat [command]

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="clean" goto clean
if "%1"=="build" goto build
if "%1"=="check" goto check
if "%1"=="upload-test" goto upload_test
if "%1"=="upload" goto upload
if "%1"=="install-dev" goto install_dev
if "%1"=="test" goto test
if "%1"=="format" goto format
if "%1"=="lint" goto lint

:help
echo Available commands:
echo   build.bat clean         Clean build artifacts
echo   build.bat build         Build package
echo   build.bat check         Check package quality
echo   build.bat upload-test   Upload to Test PyPI
echo   build.bat upload        Upload to PyPI
echo   build.bat install-dev   Install in development mode
echo   build.bat test          Run tests
echo   build.bat format        Format code
echo   build.bat lint          Run linting
goto end

:clean
echo Cleaning build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
echo Clean completed.
goto end

:build
echo Building package...
python -m build
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)
echo Build completed successfully.
goto end

:check
echo Checking package...
twine check dist/*
if %errorlevel% neq 0 (
    echo Package check failed!
    exit /b 1
)
echo Package check passed.
goto end

:upload_test
echo Uploading to Test PyPI...
twine upload --repository testpypi dist/*
goto end

:upload
echo Uploading to PyPI...
twine upload dist/*
goto end

:install_dev
echo Installing in development mode...
pip install -e .
pip install -r requirements-dev.txt
goto end

:test
echo Running tests...
python final_check.py
goto end

:format
echo Formatting code...
black ggpubpy tests examples
isort ggpubpy tests examples
goto end

:lint
echo Running linting...
flake8 ggpubpy tests examples
goto end

:end
