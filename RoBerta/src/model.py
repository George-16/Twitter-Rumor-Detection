#!/usr/bin/env python
import argparse
import torch
from pathlib import Path
from dataset import TrainDataset, DevDataset, TestDataset
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AdamW
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from sklearn.metrics import precision_recall_fscore_support


class RumourCLS(nn.Module):
    def __init__(self, pre_trained):

        super(RumourCLS, self).__init__()
        # Apply specific pre-trained model
        self.encoder = AutoModel.from_pretrained(pre_trained)
        # Set the hidden state from BERT configure
        self.cls = nn.Linear(768, 2)

    def forward(self, ids, masks):
        # Embed the text of the last hidden state
        texts_embedding = self.encoder(input_ids=ids, attention_mask=masks).last_hidden_state
        # Pick the first token
        texts_embedding = texts_embedding[:, 0, :]

        return self.cls(texts_embedding)


class RumourDetection(LightningModule):
    def __init__(self, hparams):
        # save hparams in a dict
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        # initialize the model
        super().__init__()
        self.save_hyperparameters(hparams)

        # initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pre_trained)
        self.model = RumourCLS(hparams.pre_trained)

        # initialize output path and the confusion metric
        self.output_dir = Path(hparams.output_dir)
        self.save_metrics = Path(hparams.output_dir) / 'metrics.json'
        self.save_hparams = Path(self.output_dir) / 'hparams.pkl'
        self.metrics = dict()
        self.metrics['dev'] = []
        self.confusion_metric = 'f1_score'

    def re_init(self, hparams):
        # re-initialize the model when implementing test
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        # re-initialize the model when testing
        self.save_hyperparameters(hparams)

        # re-initialize tokenizer and model when testing
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pre_trained)

        # re-initialize output path and the confusion metric when testing
        self.output_dir = Path(hparams.output_dir)
        self.save_metrics = Path(hparams.output_dir) / 'metrics.json'
        self.save_hparams = Path(self.output_dir) / 'hparams.pkl'
        self.metrics = dict()
        self.metrics['dev'] = []
        self.confusion_metric = 'f1_score'

    def configure_optimizers(self):
        # optimizer setup
        decay_indicator = ['bias', 'LayerNorm.weight']

        # parameters = dict()
        # parameters['params'] = []
        # parameters['weight_decay'] = []
        #
        # for name, parameter in self.cls.named_parameters():
        #     for no_decay_indicator in decay_indicator:
        #         if any(no_decay_indicator not in name):
        #             parameters['params'].append(parameter)
        #             parameters['weight_decay'].append(self.hparams.weight_decay)
        #         else:
        #             parameters['params'].append(parameter)
        #             parameters['weight_decay'].append(0.0)

        parameters = [
            {'params': [parameter for name, parameter in self.cls.named_parameters()
                        if not any(no_decay_indicator in name for no_decay_indicator in decay_indicator)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [parameter for name, parameter in self.cls.named_parameters()
                        if any(no_decay_indicator in name for no_decay_indicator in decay_indicator)],
             'weight_decay': 0.0}
        ]

        model_optimizer = AdamW(parameters, lr=self.hparams.learning_rate)

        # scheduler setup
        # training iteration depends on the size of dataset

        # total_steps = int((len(self.train_dataloader().dataset) / self.hparams.train_batch_size)
        #                   * self.hparams.max_epochs)
        # scheduler = get_linear_schedule_with_warmup(
        #     model_optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=total_steps)

        total_steps = int((len(self.train_dataloader().dataset) / self.hparams.train_batch_size)
                          * self.hparams.max_epochs)
        scheduler = get_cosine_schedule_with_warmup(
            model_optimizer,
            num_warmup_steps=int(self.hparams.warmup_ratio * total_steps),
            num_training_steps=total_steps)
        model_scheduler = {"scheduler": scheduler, "interval": 'step', "frequency": 1}

        return [model_optimizer], [model_scheduler]

    def training_step(self, batch, batch_idx):
        # monitor loss to check fitting problem
        logits = self.model(batch['input_ids'], batch['attn_mask'])
        labels = batch['label']
        loss = F.cross_entropy(logits, labels)
        # loss = torch.nn.CrossEntropyLoss(logits, labels, label_smoothing=0.1)

        return loss

    def validation_step(self, batch, batch_idx):
        # calculate the logits and then obtain the predictions by softmax
        # compare the predictions with labels to check the metric
        logits = self.model(batch['input_ids'], batch['attn_mask'])

        # predictions = logits.detach().cpu().numpy()
        predictions = torch.argmax(logits, dim=1).tolist()
        labels = batch['label']

        return {"predictions": predictions, "labels": labels.tolist()}

    def test_step(self, batch, batch_idx):
        # calculate the logits and then obtain the predictions by softmax
        logits = self.model(batch['input_ids'], batch['attn_mask'])

        # predictions = logits.detach().cpu().numpy()
        predictions = torch.argmax(logits, dim=1).tolist()

        return {"predictions": predictions}
        # return {"predictions": predictions, "covid_tweet": batch['covid_tweet']}

    def train_dataloader(self):
        # set up train dataloader
        dataset = TrainDataset(
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

        return dataloader

    def val_dataloader(self):
        # set up dev dataloader
        dataset = DevDataset(
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.dev_batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )
        return dataloader

    def test_dataloader(self):
        # set up test/covid dataloader
        dataset = TestDataset(
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.test_batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

        return dataloader

    def validation_epoch_end(self, outputs):
        predictions = []
        labels = []
        for output in outputs:
            for prediction, label in zip(output['predictions'], output['labels']):
                predictions.append(prediction)
                labels.append(label)

        p, r, f, _ = precision_recall_fscore_support(labels, predictions, pos_label=1, average='binary')
        # display the metric file
        self.log(self.confusion_metric, f, logger=False)

        confusion_metrics = dict()
        confusion_metrics[f'dev_precision'] = p
        confusion_metrics[f'dev_recall'] = r
        confusion_metrics[f'dev_f1_score'] = f
        self.metrics['dev'].append(confusion_metrics)

    def test_epoch_end(self, outputs):
        predictions = []
        for output in outputs:
            for prediction in output['predictions']:
                predictions.append(prediction)

        # display results
        output_path = Path(self.hparams.output_dir)
        # results_file = output_path / 'covid.predictions.csv'
        results_file = output_path / 'test.predictions.csv'

        # write the results into csv format
        with open(results_file, 'w') as w:
            w.write('Id,Predicted\n')
            for prediction_id, prediction in enumerate(predictions):
                w.write(str(prediction_id) + ',' + str(prediction) + '\n')

        # analysis covid data
        # predictions = []
        # tweets = []
        # for output in outputs:
        #     for prediction, tweet in zip(output['predictions'], output['covid_tweet']):
        #         predictions.append(prediction)
        #         tweets.append(tweet)
        #
        # output_path = Path(self.hparams.output_dir)
        # rumour_file = output_path / 'rumour.jsonl'
        # nonrumour_file = output_path / 'nonrumour.jsonl'
        #
        # rumour_writer = open(rumour_file, 'w')
        # nonrumour_writer = open(nonrumour_file, 'w')
        # for prediction, tweet in zip(predictions, tweets):
        #     if prediction == 0:
        #         nonrumour_writer.write(tweet + '\n')
        #     else:
        #         rumour_writer.write(tweet + '\n')
