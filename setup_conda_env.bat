@echo off
REM MONO-S2S Conda Environment Setup Script for Windows
REM Requires: Anaconda or Miniconda installed

echo ==========================================
echo MONO-S2S Environment Setup (Windows)
echo ==========================================

SET ENV_NAME=mono-s2s

REM Check if conda is available
where conda >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: conda not found. Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Remove existing environment if it exists
echo Checking for existing environment...
conda env list | findstr /C:"%ENV_NAME%" >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo Removing existing %ENV_NAME% environment...
    call conda env remove -n %ENV_NAME% -y
)

REM Create new conda environment with Python 3.10
echo Creating conda environment: %ENV_NAME%
call conda create -n %ENV_NAME% python=3.10 -y
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)

REM Activate environment
echo Activating environment...
call conda activate %ENV_NAME%

REM Install PyTorch with CUDA support for Windows
echo Installing PyTorch with CUDA support...
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
IF %ERRORLEVEL% NEQ 0 (
    echo WARNING: Failed to install CUDA version, trying CPU-only...
    call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
)

REM Install core scientific computing packages
echo Installing core scientific packages...
call conda install -y numpy scipy pandas matplotlib

REM Install Jupyter and related tools
echo Installing Jupyter...
call conda install -y jupyter jupyterlab ipykernel notebook

REM Install additional packages via pip
echo Installing additional packages via pip...
call pip install --upgrade pip
call pip install datasets
call pip install transformers
call pip install accelerate
call pip install tqdm
call pip install scikit-learn
call pip install ipywidgets

REM Enable ipywidgets
call jupyter nbextension enable --py widgetsnbextension --sys-prefix

REM Verify PyTorch CUDA availability
echo.
echo ==========================================
echo Verifying PyTorch Installation...
echo ==========================================
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

REM Create project directory structure
echo.
echo Creating project directory structure...
if not exist "data\checkpoints" mkdir data\checkpoints
if not exist "data\tokenizer" mkdir data\tokenizer
if not exist "results" mkdir results
if not exist "logs" mkdir logs

REM Create local config file
echo Creating local configuration...
(
echo """
echo Local configuration to replace Google Colab Drive paths.
echo Import this at the start of your notebook with:
echo     from local_config import *
echo """
echo import os
echo.
echo # Base paths
echo PROJECT_ROOT = os.path.dirname^(os.path.abspath^(__file__^)^)
echo DATA_PATH = os.path.join^(PROJECT_ROOT, 'data'^)
echo CHECKPOINT_PATH = os.path.join^(DATA_PATH, 'checkpoints'^)
echo TOKENIZER_PATH = os.path.join^(DATA_PATH, 'tokenizer', 'tokenizer_v4.json'^)
echo RESULTS_PATH = os.path.join^(PROJECT_ROOT, 'results'^)
echo LOGS_PATH = os.path.join^(PROJECT_ROOT, 'logs'^)
echo.
echo # Create directories if they don't exist
echo os.makedirs^(CHECKPOINT_PATH, exist_ok=True^)
echo os.makedirs^(os.path.dirname^(TOKENIZER_PATH^), exist_ok=True^)
echo os.makedirs^(RESULTS_PATH, exist_ok=True^)
echo os.makedirs^(LOGS_PATH, exist_ok=True^)
echo.
echo print^(f"✓ Local paths configured:"^)
echo print^(f"  - Checkpoints: {CHECKPOINT_PATH}"^)
echo print^(f"  - Tokenizer: {TOKENIZER_PATH}"^)
echo print^(f"  - Results: {RESULTS_PATH}"^)
echo print^(f"  - Logs: {LOGS_PATH}"^)
) > local_config.py

REM Create requirements.txt
echo Creating requirements.txt for reference...
(
echo torch^>=2.1.0
echo torchvision^>=0.16.0
echo torchaudio^>=2.1.0
echo numpy^>=1.24.0
echo scipy^>=1.11.0
echo pandas^>=2.0.0
echo matplotlib^>=3.7.0
echo jupyter^>=1.0.0
echo jupyterlab^>=4.0.0
echo datasets^>=2.14.0
echo transformers^>=4.30.0
echo accelerate^>=0.20.0
echo tqdm^>=4.65.0
echo scikit-learn^>=1.3.0
echo ipywidgets^>=8.0.0
) > requirements.txt

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To use this environment:
echo   1. Activate: conda activate %ENV_NAME%
echo   2. Start Jupyter Lab: jupyter lab
echo   3. Open Mono_S2S_v1_2.ipynb
echo.
echo IMPORTANT NOTES:
echo   - Replace Google Colab drive.mount() calls with:
echo       from local_config import *
echo   - Replace DRIVE_PATH with DATA_PATH
echo   - The notebook expects CUDA. If unavailable, training will be slow.
echo.
echo To verify CUDA is working:
echo   conda activate %ENV_NAME%
echo   python -c "import torch; print(torch.cuda.is_available())"
echo.
echo ==========================================
pause

