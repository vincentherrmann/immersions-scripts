import sys
import torch
import subprocess

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
except:
    sys.path.append('../constrastive-predictive-coding-audio')
    from audio_dataset import *
    from audio_model import *
    from contrastive_estimation_training import *

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("using device", dev)

name = "model_2019-01-07_run_0"

encoding_size = 512
ar_code_size = 256
visible_steps = 128
prediction_steps = 12
batch_size = 64
lr = 1e-4

training_set_location = '../data/MelodicProgressiveHouseMix_train'
validation_set_location = '../data/MelodicProgressiveHouseMix_test'
logs_location = '../logs'
snapshot_location = '../snapshots'
ngrok = '../misc/ngrok'

encoder_params = encoder_default_dict
encoder_params["channel_count"] = [encoding_size for _ in range(5)]
encoder = AudioEncoder(encoder_params)

ar_model = AudioGRUModel(input_size=encoding_size, hidden_size=ar_code_size)
pc_model = AudioPredictiveCodingModel(encoder=encoder,
                                      autoregressive_model=ar_model,
                                      enc_size=encoding_size,
                                      ar_size=ar_code_size,
                                      prediction_steps=prediction_steps)

pc_model.to(dev)
print("number of parameters:", pc_model.parameter_count())
print("receptive field:", encoder.receptive_field)

gcs_manager = GCSManager('pytorch-wavenet', 'immersions')
snapshot_manager = SnapshotManager(pc_model,
                                   gcs_manager,
                                   name=name,
                                   snapshot_location=snapshot_location,
                                   logs_location=logs_location,
                                   gcs_snapshot_location='snapshots',
                                   gcs_logs_location='logs')

prediction_steps = pc_model.prediction_steps
item_length = encoder.receptive_field + (visible_steps + prediction_steps - 1) * encoder.downsampling_factor
visible_length = encoder.receptive_field + (visible_steps - 1) * encoder.downsampling_factor
prediction_length = encoder.receptive_field + (prediction_steps - 1) * encoder.downsampling_factor

print("item length:", item_length)

dataset = AudioDataset(training_set_location,
                       item_length=item_length,
                       unique_length=prediction_steps * encoder.downsampling_factor)
print("dataset length:", len(dataset))
validation_set = AudioDataset(validation_set_location,
                              item_length=item_length,
                              unique_length=prediction_steps * encoder.downsampling_factor)
print("validation set length:", len(validation_set))

dataset.dummy_load = False
trainer = ContrastiveEstimationTrainer(model=pc_model,
                                       dataset=dataset,
                                       validation_set=validation_set,
                                       visible_length=visible_length,
                                       prediction_length=prediction_length,
                                       device=dev,
                                       regularization=1.)


def background_func(current_step):
    snapshot_manager.upload_latest_files()


def dummy_validation_function():
    losses, accuracies, mean_score = trainer.validate(batch_size=batch_size, num_workers=4, max_steps=300)
    print(losses, accuracies, mean_score)
    logger.writer.add_scalar("score mean", mean_score, trainer.training_step)
    for step in range(losses.shape[0]):
        logger.writer.add_scalar("validation loss/step " + str(step),
                                 losses[step].item(),
                                 trainer.training_step)
        logger.writer.add_scalar("validation accuracy/step " + str(step),
                                 accuracies[step].item(),
                                 trainer.training_step)
    return torch.mean(losses).item(), torch.mean(accuracies).item()


logger = TensorboardLogger(log_interval=20,
                           validation_function=dummy_validation_function,
                           validation_interval=1000,
                           log_directory=snapshot_manager.current_tb_location,
                           snapshot_function=snapshot_manager.make_snapshot,
                           snapshot_interval=5000,
                           background_function=background_func,
                           background_interval=5000)
trainer.logger = logger

print("start tensorboard")
subprocess.check_call(['tensorboard', '--logdir', logs_location, '--host', '0.0.0.0', '--port', '6007', '&'])
subprocess.check_call([ngrok, 'http', '6007', '&'])

print("first validation...")
trainer.training_step = 0
logger.validate(trainer.training_step)

print("start training")
trainer.train(batch_size=batch_size, epochs=100, lr=lr, continue_training_at_step=0, num_workers=4)





