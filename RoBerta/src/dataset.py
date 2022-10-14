#!/usr/bin/env python
import json
import os
import time
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(
            self,
            tokenizer,
            max_length
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        # # data type with one json file corresponding to one event
        # train_tweet_labels = open('data/train.label.txt', 'r')
        # self.train_texts = []
        # self.train_labels = []
        # idx = 0
        # for tweet_label in train_tweet_labels.readlines():
        #     tweet_list = []
        #     idx += 1
        #     train_path = 'data/train/' + str(idx) + '.json'
        #     if os.path.exists(train_path):
        #         tweet_list.append(json.load(open(train_path, 'r')))
        #     else:
        #         continue
        #     tweet_list = sorted(
        #         tweet_list, key=lambda x: time.mktime(time.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')))
        #     self.train_texts.append(tweet_list)
        #     if tweet_label.strip() == 'rumour':
        #         tag = 1
        #     else:
        #         tag = 0
        #     self.train_labels.append(tag)

        # data type with one json file corresponding to one tweet
        # read tweets id and labels
        train_tweet_ids = open('data/train.data.txt', 'r')
        train_tweet_labels = open('data/train.label.txt', 'r')
        self.train_texts = []
        self.train_labels = []
        # concat each id with its label
        for tweet_ids, tweet_label in zip(train_tweet_ids.readlines(), train_tweet_labels.readlines()):
            tweet_ids_list = tweet_ids.strip().split(',')
            tweet_list = []
            if not os.path.exists('data/train/' + tweet_ids_list[0] + '.json'):
                continue
            for tweet_id in tweet_ids_list:
                # read json file one by one
                train_path = 'data/train/' + tweet_id + '.json'
                if os.path.exists(train_path):
                    tweet_list.append(json.load(open(train_path, 'r')))
            # sort according to time according to particular token types
            # the purpose is to guarantee the logic of tweets order
            tweet_list = sorted(
                tweet_list, key=lambda x: time.mktime(time.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')))
            self.train_texts.append(tweet_list)
            # convert label to numerical label
            if tweet_label.strip() == 'rumour':
                tag = 1
            else:
                tag = 0
            self.train_labels.append(tag)

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        return self.train_texts[index], self.train_labels[index]

    def collate_fn(self, dataset):
        texts = []
        labels = []
        for tweets, label in dataset:
            text = []
            for tweet in tweets:
                # read out the text from each tweet and preprocess it
                text.append(preprocess(tweet['text']))
            texts.append(self.tokenizer.sep_token.join(text))
            labels.append(label)

        # tokenize the texts
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            return_tensors='pt',
            truncation=True
        )

        # prepare inputs
        inputs = {}
        inputs['input_ids'] = encoded.input_ids
        inputs['attn_mask'] = encoded.attention_mask
        inputs['label'] = torch.LongTensor(labels)

        return inputs


class DevDataset(Dataset):

    def __init__(
            self,
            tokenizer,
            max_length
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        # # data type with one json file corresponding to one event
        # dev_tweet_labels = open('data/dev.label.txt', 'r')
        # self.dev_texts = []
        # self.dev_labels = []
        # idx = 0
        # for tweet_label in dev_tweet_labels.readlines():
        #     tweet_list = []
        #     idx += 1
        #     dev_path = 'data/dev/' + str(idx) + '.json'
        #     if os.path.exists(dev_path):
        #         tweet_list.append(json.load(open(dev_path, 'r')))
        #     else:
        #         continue
        #     tweet_list = sorted(
        #         tweet_list, key=lambda x: time.mktime(time.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')))
        #     self.dev_texts.append(tweet_list)
        #     if tweet_label.strip() == 'rumour':
        #         tag = 1
        #     else:
        #         tag = 0
        #     self.dev_labels.append(tag)

        # data type with one json file corresponding to one tweet
        # basic logic above
        # read tweets id and labels
        dev_tweet_ids = open('data/dev.data.txt', 'r')
        dev_tweet_labels = open('data/dev.label.txt', 'r')
        self.dev_texts = []
        self.dev_labels = []
        # concat each id with its label
        for tweet_ids, tweet_label in zip(dev_tweet_ids.readlines(), dev_tweet_labels.readlines()):
            tweet_ids_list = tweet_ids.strip().split(",")
            tweet_list = []
            if not os.path.exists('data/dev/' + tweet_ids_list[0] + '.json'):
                continue
            for tweet_id in tweet_ids_list:
                # read json file one by one
                dev_path = 'data/dev/' + tweet_id + '.json'
                if os.path.exists(dev_path):
                    tweet_list.append(json.load(open(dev_path, 'r')))
            # sort according to time according to particular token types
            # the purpose is to guarantee the logic of tweets order
            tweet_list = sorted(
                tweet_list, key=lambda x: time.mktime(time.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')))
            self.dev_texts.append(tweet_list)
            # convert label to numerical label
            if tweet_label.strip() == 'rumour':
                tag = 1
            else:
                tag = 0
            self.dev_labels.append(tag)

    def __len__(self):
        return len(self.dev_labels)

    def __getitem__(self, index):
        return self.dev_texts[index], self.dev_labels[index]

    def collate_fn(self, dataset):
        # basic logic above
        texts = []
        labels = []
        for tweets, label in dataset:
            text = []
            for tweet in tweets:
                # read out the text from each tweet and preprocess it
                text.append(preprocess(tweet['text']))
            texts.append(self.tokenizer.sep_token.join(text))
            labels.append(label)

        # tokenize the texts
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            return_tensors='pt',
            truncation=True
        )

        # prepare inputs
        inputs = {}
        inputs['input_ids'] = encoded.input_ids
        inputs['attn_mask'] = encoded.attention_mask
        inputs['label'] = torch.LongTensor(labels)

        return inputs


class TestDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        max_length
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        # # data type with one json file corresponding to one event
        # self.test_texts = []
        # for idx in range(557):
        #     tweet_list = []
        #     test_path = 'data/test/' + str(idx) + '.json'
        #     if os.path.exists(test_path):
        #         tweet_list.append(json.load(open(test_path, 'r')))
        #     else:
        #         continue
        #     tweet_list = sorted(
        #         tweet_list, key=lambda x: time.mktime(time.strptime(x['created_at'], '%a %b %d %H:%M:%S +0000 %Y')))
        #     self.test_texts.append(tweet_list)

        # # covid analysis
        # for idx in range(15955):
        #     tweet_list = []
        #     test_path = 'data/analysis/' + str(idx) + '.json'
        #     if os.path.exists(test_path):
        #         tweet_list.append(json.load(open(test_path, 'r')))
        #     else:
        #         continue
        #     tweet_list = sorted(
        #         tweet_list, key=lambda x: time.mktime(time.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')))
        #     self.test_texts.append(tweet_list)

        # data type with one json file corresponding to one tweet
        # basic logic above
        # read tweets id and labels
        test_tweet_ids = open('data/test.data.txt', 'r')
        # test_tweet_ids = open("data/covid.data.txt", "r")
        self.test_texts = []
        for tweet_ids in test_tweet_ids.readlines():
            tweet_ids_list = tweet_ids.strip().split(",")
            tweet_list = []
            if not os.path.exists('data/test' + tweet_ids_list[0] + '.json'):
                continue
            # if not os.path.exists('data/analysis/' + tweet_ids_list[0] + '.json'):
            #     continue
            for tweet_id in tweet_ids_list:
                # read json file one by one
                test_path = 'data/test/' + tweet_id + '.json'
                # test_path = 'data/analysis_tweet/' + tweet_id + '.json'
                if os.path.exists(test_path):
                    tweet_list.append(json.load(open(test_path, 'r')))
            # sort according to time according to particular token types
            # the purpose is to guarantee the logic of tweets order
            tweet_list = sorted(
                tweet_list, key=lambda x: time.mktime(time.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')))
            # tweet_list = sorted(
            # tweet_list, key=lambda x: time.mktime(time.strptime(x["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')))
            self.test_texts.append(tweet_list)

    def __len__(self):
        return len(self.test_texts)

    def __getitem__(self, index):
        return self.test_texts[index]

    def collate_fn(self, dataset):
        # basic logic above
        texts = []
        for tweets, label in dataset:
            text = []
            for tweet in tweets:
                # read out the text from each tweet and preprocess it
                text.append(preprocess(tweet['text']))
            texts.append(self.tokenizer.sep_token.join(text))

        # tokenize the texts
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            return_tensors='pt',
            truncation=True
        )

        # prepare inputs
        inputs = {}
        inputs['input_ids'] = encoded.input_ids
        inputs['attn_mask'] = encoded.attention_mask

        # covid_tweets = []
        # for covid_tweet in dataset:
        #     covid_tweets.append(json.dumps(covid_tweet[0]))
        # inputs['covid_tweet'] = covid_tweet

        return inputs


# retrieve from huggingface
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
