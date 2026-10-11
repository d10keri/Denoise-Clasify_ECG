# Environment Configuration
This project MUST strictly use the existing Conda environment 'ecg'.
- Python path: C:\Users\vietc\.conda\envs\ecg\python.exe
- Pip path: C:\Users\vietc\.conda\envs\ecg\Scripts\pip.exe

Rules for Agent:
1. ALWAYS execute Python files using: & "C:\Users\vietc\.conda\envs\ecg\python.exe" <script.py>
2. All packages (PyTorch with CUDA, wfdb, scipy, scikit-learn, PyWavelets, etc.) are ALREADY installed. NEVER execute `pip install` under any circumstances.
3. NEVER call system python or attempt to create any local virtualenv.
