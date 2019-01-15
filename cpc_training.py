import argparse
import os
import os.path
import glob
import sys
import torch
import subprocess
import datetime
import torch
import threading
from torch import autograd

parser = argparse.ArgumentParser(description='Contrastive Predictive Coding Training')
parser.add_argument('--encoding-size', default=512, type=int)
parser.add_argument('--ar-code-size', default=256, type=int)
parser.add_argument('--visible-steps', default=128, type=int)
parser.add_argument('--prediction-steps', default=12, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--training-set', default='../data/MelodicProgressiveHouseMix_train', type=str)
parser.add_argument('--validation-set', default='../data/MelodicProgressiveHouseMix_test', type=str)
parser.add_argument('--task-set', default='../data/MelodicProgressiveHouse_test', type=str)
parser.add_argument('--logs-dir', default='../logs', type=str)
parser.add_argument('--snapshots-dir', default='../snapshots', type=str)
parser.add_argument('--name', default='model_' + datetime.datetime.today().strftime('%Y-%m-%d') + '_run_0', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--detect-anomalies', default=False, type=bool)
parser.add_argument('--regularization', default=1.0, type=float)
parser.add_argument('--ar-model', default='gru', type=str)

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
    from attention_model import *
    from contrastive_estimation_training import *
except:
    sys.path.append('../constrastive-predictive-coding-audio')
    from audio_dataset import *
    from audio_model import *
    from attention_model import *
    from contrastive_estimation_training import *

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("using device", dev)

class CPCLogger(TensorboardLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_meter = AverageMeter()

    def log_loss(self, current_step):
        self.writer.add_scalar('loss', self.loss_meter.avg, current_step)
        self.loss_meter.reset()
        self.writer.add_scalar('max_score', self.score_meter.max, current_step)
        self.score_meter.reset()


def main():
    args = parser.parse_args()

    encoder_params = encoder_default_dict
    encoder_params["channel_count"] = [args.encoding_size for _ in range(5)]
    encoder = AudioEncoder(encoder_params)

    if args.ar_model == 'gru' or args.ar_model == 'GRU':
        ar_model = AudioGRUModel(input_size=args.encoding_size,
                                 hidden_size=args.ar_code_size)
    elif args.ar_model == 'transformer' or args.ar_model == 'attention':
        ar_model = AttentionModel(channels=args.encoding_size,
                                  output_size=args.ar_code_size,
                                  num_layers=2,
                                  num_heads=8,
                                  feedforward_size=args.encoding_size,
                                  seq_length=args.visible_steps)
    else:
        raise Exception('no autoregressive mode named ' + args.ar_model)
    pc_model = AudioPredictiveCodingModel(encoder=encoder,
                                          autoregressive_model=ar_model,
                                          enc_size=args.encoding_size,
                                          ar_size=args.ar_code_size,
                                          prediction_steps=args.prediction_steps)

    pc_model.to(dev)
    print("number of parameters:", pc_model.parameter_count())
    print("receptive field:", encoder.receptive_field)

    gcs_manager = GCSManager('pytorch-wavenet', 'immersions')
    snapshot_manager = SnapshotManager(pc_model,
                                       gcs_manager,
                                       name=args.name,
                                       snapshot_location=args.snapshots_dir,
                                       logs_location=args.logs_dir,
                                       gcs_snapshot_location='snapshots',
                                       gcs_logs_location='logs')

    continue_training_at_step = 0
    try:
        pc_model, newest_snapshot = snapshot_manager.load_latest_snapshot()
        encoder = pc_model.encoder
        ar_model = pc_model.autoregressive_model
        pc_model.to(dev)
        continue_training_at_step = int(newest_snapshot.split('_')[-1])
        print("loaded", newest_snapshot)
        print("continue training at step", continue_training_at_step)
    except:
        print("no previous snapshot found, starting training from scratch")

    prediction_steps = pc_model.prediction_steps
    item_length = encoder.receptive_field + (args.visible_steps + prediction_steps - 1) * encoder.downsampling_factor
    visible_length = encoder.receptive_field + (args.visible_steps - 1) * encoder.downsampling_factor
    prediction_length = encoder.receptive_field + (prediction_steps - 1) * encoder.downsampling_factor

    print("item length:", item_length)

    dataset = AudioDataset(args.training_set,
                           item_length=item_length,
                           unique_length=prediction_steps * encoder.downsampling_factor)
    print("dataset length:", len(dataset))
    validation_set = AudioDataset(args.validation_set,
                                  item_length=item_length,
                                  unique_length=prediction_steps * encoder.downsampling_factor)
    print("validation set length:", len(validation_set))
    task_set = AudioTestingDataset(args.task_set,
                                   item_length=item_length)
    print("task set length:", len(task_set))

    dataset.dummy_load = False
    trainer = ContrastiveEstimationTrainer(model=pc_model,
                                           dataset=dataset,
                                           validation_set=validation_set,
                                           test_task_set=task_set,
                                           visible_length=visible_length,
                                           prediction_length=prediction_length,
                                           device=dev,
                                           regularization=args.regularization)
    #task_thread = threading.Thread()
    #print(task_thread)

    def background_func(current_step):
        snapshot_manager.upload_latest_files()

    def dummy_validation_function():
        #if not task_thread.isAlive():
        print("start task test")
        task_data, task_labels = trainer.calc_test_task_data(batch_size=args.batch_size, num_workers=4)
        task_thread = threading.Thread(target=task_function,
                                       args=(task_data, task_labels, trainer.training_step),
                                       daemon=False)
        task_thread.start()

        losses, accuracies, mean_score = trainer.validate(batch_size=args.batch_size, num_workers=4, max_steps=300)
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


    def task_function(task_data, task_labels, step):
        task_accuracy = trainer.test_task(task_data, task_labels)
        logger.writer.add_scalar("task accuracy", task_accuracy, step)


    logger = CPCLogger(log_interval=20,
                       validation_function=dummy_validation_function,
                       validation_interval=1000,
                       log_directory=snapshot_manager.current_tb_location,
                       snapshot_function=snapshot_manager.make_snapshot,
                       snapshot_interval=5000,
                       background_function=background_func,
                       background_interval=5000)
    trainer.logger = logger

    if continue_training_at_step == 0:
        print("first validation...")
        trainer.training_step = 0
        logger.validate(trainer.training_step)

    print("start training")
    if args.detect_anomalies:
        print("run with anomaly detection")
        print("start training")
        with autograd.detect_anomaly():
            try:
                trainer.train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                              continue_training_at_step=continue_training_at_step,
                              num_workers=4)
            except Exception as e:
                print(e)
                snapshot_manager.make_snapshot('error')
                print("error at step", trainer.training_step)
    else:
        print("start training")
        trainer.train(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                      continue_training_at_step=continue_training_at_step,
                      num_workers=4)


if __name__ == '__main__':
    main()


