import sys
import os
print('Python executable:', sys.executable)
print('Python version:', sys.version)
print('CWD:', os.getcwd())
modules = ['tkinter', 'requests', 'docx2pdf', 'docx', 'reportlab']
for m in modules:
    try:
        __import__(m)
        print(m, 'OK')
    except Exception as e:
        print(m, 'ERROR:', type(e).__name__, str(e))

# Check if running in interactive/GUI-capable session
print('Has DISPLAY (env):', os.environ.get('DISPLAY'))
# Check tkinter available and basic create
try:
    import tkinter as tk
    root = tk.Tk()
    root.update_idletasks()
    print('Tkinter root created: OK')
    root.destroy()
except Exception as e:
    print('Tkinter runtime ERROR:', type(e).__name__, e)
