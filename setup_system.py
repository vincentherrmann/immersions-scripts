import subprocess
import sys


def install(package):
    try:
        if type(package) is str:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user"] + package)
        print("installed", package)
    except:
        print("error while installing", package)

try:
    import librosa as lr
except:
    install('librosa')

try:
    import matplotlib.pyplot as plt
except:
    install("matplotlib")

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except:
    install("PyDrive")

try:
    import tensorboardX
except:
    install("tensorboardX")

try:
    import torchaudio
except:
    subprocess.check_call(["sudo", "apt-get", "install", "sox", "libsox-dev", "libsox-fmt-all"])
    install("git+git://github.com/pytorch/audio")

print("all packages installed")

subprocess.check_call(["cd", "../;",
                       "git", "clone", "https://github.com/vincentherrmann/pytorch-utilities.git"])
subprocess.check_call(["cd", "../;",
                       "git", "clone", "https://github.com/vincentherrmann/constrastive-predictive-coding-audio.git"])

print("repositories downloaded")


