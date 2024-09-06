import subprocess
import torch

def get_nvidia_smi_output():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

smi_output = get_nvidia_smi_output()
print(smi_output)
print(torch.version.cuda)