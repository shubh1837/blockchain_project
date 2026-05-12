@echo off
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Starting Model Training on GPU...
python scripts\train_model.py --subset 0 --epochs 5 --batch-size 8
echo.
pause
