import subprocess
import sys


def install(package):
    if type(package) is str:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + package)
    print("installed", package)

try:
    import torch
except:
    install(['torch', 'torchvision'])
    import torch

try:
    import librosa as lr
except:
    install('librosa')
    import librosa as lr

try:
    import matplotlib.pyplot as plt
except:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except:
    install("PyDrive")
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive

try:
    import tensorboardX
except:
    install("tensorboardX")
    import tensorboardX

try:
    import torchaudio
except:
    subprocess.check_call(["sudo", "apt-get", "install", "sox", "libsox-dev", "libsox-fmt-all"])
    install("git+git://github.com/pytorch/audio")
    import torchaudio

print("all packages installed")

subprocess.check_call(["cd", "../;",
                       "git", "clone", "https://github.com/vincentherrmann/pytorch-utilities.git"])
subprocess.check_call(["cd", "../;",
                       "git", "clone", "https://github.com/vincentherrmann/constrastive-predictive-coding-audio.git"])

print("repositories downloaded")


