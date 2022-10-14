#!/usr/bin/env python
import json
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only


def save_json_file(metrics, path, **json_dump_kwargs):
    # save the output of confusion metrics in each batch of epochs
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=5, **json_dump_kwargs)


class DisplayCallback(Callback):
    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        print(' -----Training completes!-----')

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        save_json_file(pl_module.metrics, pl_module.save_metrics)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        print('-----Testing completes!-----')
        save_json_file(pl_module.metrics, pl_module.save_metrics)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        print('-----Training starts!-----')

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        print('-----Testing starts!-----')
