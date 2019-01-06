import sys
import torch

try:
    from colab_utilities import GCSManager, SnapshotManager
    from train_logging import *
except:
    sys.path.append('../pytorch-utilities')
    from colab_utilities import GCSManager, SnapshotManager
    from train_logging import *

try:
    from audio_dataset import *
    from audio_model import *
    from contrastive_estimation_training import *
except
    sys.path.append('../constrastive-predictive-coding-audio')
    from audio_dataset import *
    from audio_model import *
    from contrastive_estimation_training import *


sys.path.append('audio')
import torchaudio

