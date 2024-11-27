@echo off

:: Run the first script
python models-abo\svm-randomtree-xgboost\train.py
if %ERRORLEVEL% NEQ 0 (
    echo Code 1 encountered an error. Exiting.
    exit /b %ERRORLEVEL%
)
echo Code 1 executed successfully. Starting Code 2...

:: Run the second script
python models-abo\pointnet\train.py
if %ERRORLEVEL% NEQ 0 (
    echo Code 2 encountered an error.
) else (
    echo Code 2 executed successfully.
)
