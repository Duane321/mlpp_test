import subprocess
import os

"""
STILL NEED TO CORRECT THE INTERNAL REFERENCES ISSUE
"""

output_folder = 'Exercises'
os.chdir(output_folder)

file_names = [e for e in os.listdir() if e.endswith('.tex')]

def run(cmd):
    print('executing {}'.format(cmd))	
    proc = subprocess.Popen(cmd)
    proc.communicate()
    retcode = proc.returncode
    if (not retcode == 0):
        #raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd))) 
        msg = 'Error {} executing command: {}'.format(retcode, ' '.join(cmd))
        return msg
    else:
        return 'ok'

for key in file_names:
    # nm = '{}/{}'.format(output_folder, key)
    nm = '{}'.format(key)
    # Compile twice to get cross references right
    cmd = ['pdflatex', '-interaction', 'nonstopmode', nm]
    msg = run(cmd)
    if not(msg == 'ok'):
        raise ValueError(msg)
    msg = run(cmd)

    # # Try to create bibliography. Will fail if there are no citations.
    # # If succeeds, recompile.
    # cmd = ['bibtex', nm]
    # msg = run(cmd)
    # if (msg == 'ok'):
    #     cmd = ['pdflatex', '-interaction', 'nonstopmode', nm]
    #     run(cmd)
    #     run(cmd)