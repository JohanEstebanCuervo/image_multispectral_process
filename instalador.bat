winget install Python.Python.3.10
"C:\Users\%username%\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -U pip virtualenv
"C:\Users\%username%\AppData\Local\Programs\Python\Python310\python.exe" -m virtualenv venv
call venv\Scripts\activate.bat
pip install -r data\requirements.txt
md results results\errors results\images results\models
md imgs
pause