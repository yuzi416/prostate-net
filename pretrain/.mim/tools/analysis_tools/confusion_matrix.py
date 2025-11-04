# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
import shutil
import logging
import mmengine
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

from mmpretrain.evaluation import ConfusionMatrix
from mmpretrain.registry import DATASETS
from mmpretrain.utils import register_all_modules


# 手动设置日志关闭函数
def close_log_handlers(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def main():
    # 配置文件路径和权重文件路径硬编码
    config_path = 'E:\mmpretrain\work_dirs/res2net50-w14-s8_8xb32_in1k/20241013_160614/vis_data\config.py'
    checkpoint_path = 'E:/mmpretrain/work_dirs/res2net50-w14-s8_8xb32_in1k/epoch_50.pth'
    output_dir = 'E:/mmpretrain/work_dirs/result/'

    # register all modules in mmpretrain into the registries
    register_all_modules(init_default_scope=False)

    # 加载配置文件
    cfg = Config.fromfile(config_path)

    # 设置工作目录
    cfg.work_dir = 'E:/mmpretrain/work_dir'
    print(f'Configured work directory: {cfg.work_dir}')  # 检查工作目录是否被正确设置

    logger = logging.getLogger('mmengine')

    if checkpoint_path.endswith('.pth'):
        # 设置混淆矩阵作为评估指标
        cfg.test_evaluator = dict(type='ConfusionMatrix')

        # 加载权重文件
        cfg.load_from = checkpoint_path

        # 检查是否正确设置了工作目录
        print(f'Before Runner init, work_dir: {cfg.work_dir}')

        # 使用配置文件运行评估
        runner = Runner.from_cfg(cfg)

        # 再次检查工作目录，确保不会被覆盖
        print(f'After Runner init, work_dir: {cfg.work_dir}')

        classes = runner.test_loop.dataloader.dataset.metainfo.get('classes')
        cm = runner.test()['confusion_matrix/result']
    else:
        # 如果是预测结果文件 (.pkl)
        predictions = mmengine.load(checkpoint_path)
        evaluator = Evaluator(ConfusionMatrix())
        metrics = evaluator.offline_evaluate(predictions, None)
        cm = metrics['confusion_matrix/result']
        try:
            # 尝试构建数据集
            dataset = DATASETS.build({
                **cfg.test_dataloader.dataset, 'pipeline': []
            })
            classes = dataset.metainfo.get('classes')
        except Exception:
            classes = None

    # 保存混淆矩阵
    out_path = os.path.join(output_dir, 'confusion_matrix.pkl')
    mmengine.dump(cm, out_path)

    # 显示或保存混淆矩阵图片
    show = False  # 不显示
    show_path = os.path.join(output_dir, 'confusion_matrix.png')  # 保存路径
    fig = ConfusionMatrix.plot(
        cm,
        show=show,
        classes=classes,
        include_values=True,  # 显示数值
        cmap='Blues'
    )
    fig.savefig(show_path)
    print(f'The confusion matrix is saved at {show_path}.')


if __name__ == '__main__':
    main()
