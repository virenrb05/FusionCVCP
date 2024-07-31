import faulthandler
import torch
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import TensorBoardLogger
import os
from det3d.torchie import Config
import torch
from det3d.models.detectors.modelmodule import CPModel
from det3d.datasets import build_dataset
from torch.utils.data import DataLoader
from det3d.models import build_detector
from det3d.torchie.parallel import collate_kitti
from torch.utils.data import SequentialSampler

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    cfg = Config.fromfile(
        '/home/vxm240030/CenterPoint/configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py')

    faulthandler.enable()
    torch.cuda.empty_cache()

    logger = TensorBoardLogger(
        save_dir=cfg.work_dir,
        name='train',
        default_hp_metric=False,
    )

    hyperparameters = {
        'epochs': 10,
        'batch_size': 1,
        'lr': 0.001,
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'weight_decay': 0.01,
        'num_workers': 2,
    }

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    modelmodule = CPModel(model)

    dataset = build_dataset(cfg.data.train)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=hyperparameters['batch_size'],
        sampler=SequentialSampler(dataset),
        num_workers=hyperparameters['num_workers'],
        collate_fn=collate_kitti,
        pin_memory=True,
    )

    logger.log_hyperparams(hyperparameters)

    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True,
        log_weight_decay=True)

    checkpointer = ModelCheckpoint(
        dirpath=Path(logger.log_dir) / 'checkpoints',
        filename='{epoch}-{step}',
        verbose=True,
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=True,
        every_n_epochs=1,
    )

    model_summary = ModelSummary(max_depth=4)

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        num_nodes=1,
        max_epochs=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=logger,
        log_every_n_steps=10,
        callbacks=[checkpointer, lr_monitor, model_summary],
        limit_test_batches=1.0
    )

    print('Hyperparameters:', hyperparameters)
    print('Logger version:', logger.version)
    print('Log dir:', logger.log_dir)

    trainer.test(
        model=modelmodule,
        dataloaders=data_loader,
        ckpt_path='/home/vxm240030/CenterPoint/work_dirs/nusc_centerpoint_pp_02voxel_two_pfn_10sweep/train/version_16/checkpoints/epoch=39-step=34320.ckpt'
    )

    print('\n=========================')


if __name__ == '__main__':
    main()
