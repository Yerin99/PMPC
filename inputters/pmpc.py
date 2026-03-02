# coding=utf-8
"""
PMPC Inputter - 데이터 처리

strat inputter를 확장하여 situation 정보 포함
"""

import json
import tqdm
import torch
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
import random
from functools import partial
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from inputters.inputter_utils import _norm, BucketSampler, BucketingDataLoader, DistributedBucketingDataLoader
from .PARAMS import GOLDEN_TRUTH


class Inputter(object):
    def __init__(self):
        # prepare
        self.convert_data_to_inputs = convert_data_to_inputs
        self.convert_inputs_to_features = convert_inputs_to_features
        
        # train
        self.train_sampler = BucketSampler
        self.train_dataset = FeatureDataset
        self.train_dataloader = BucketingDataLoader
        self.train_distributed_dataloader = DistributedBucketingDataLoader
        
        # valid
        self.valid_dataloader = DynamicBatchingLoader
        
        # infer
        self.prepare_infer_batch = prepare_infer_batch
        self.infer_dataloader = get_infer_batch


class InputFeatures(object):
    """PMPC Input Features with situation"""
    def __init__(
        self,
        input_ids,
        decoder_input_ids, 
        labels,
        situation_ids=None,  # PMPC: situation tokens
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)
        
        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels
        
        # PMPC specific
        self.situation_ids = situation_ids if situation_ids else []
        self.situation_length = len(self.situation_ids) if situation_ids else 0

        self.input_len = self.input_length + self.decoder_input_length


def featurize(
    bos, eos,
    context, max_input_length,
    response, max_decoder_input_length, 
    strat_id,
    situation=None, max_situation_length=50,  # PMPC: situation
):
    """Feature 생성 - situation 포함"""
    context = [c + [eos] for c in context]
    input_ids = sum(context, [])[:-1]
    input_ids = input_ids[-max_input_length:]
    
    labels = ([strat_id] + response + [eos])[:max_decoder_input_length + 1]
    decoder_input_ids = [bos] + labels[:-1]
    
    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]
    
    # PMPC: situation tokens
    situation_ids = None
    if situation is not None:
        situation_ids = situation[:max_situation_length]

    return InputFeatures(
        input_ids,
        decoder_input_ids, labels,
        situation_ids=situation_ids,
    )


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    """데이터 → 입력 변환 - situation 포함"""
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))
    
    dialog = data['dialog']
    situation = data.get('situation', '')  # PMPC: situation
    
    # Tokenize situation
    situation_tokens = process(situation) if situation else []
    
    inputs = []
    context = []
    
    for i in range(len(dialog)):
        text = _norm(dialog[i]['text'])
        text = process(text)
        
        if dialog[i]['speaker'] == 'sys':
            strat_id = process('[' + dialog[i]['strategy'] + ']')
            assert len(strat_id) == 1
            strat_id = strat_id[0]
        
        if i > 0 and dialog[i]['speaker'] == 'sys':
            res = {
                'context': context.copy(),
                'response': text,
                'strat_id': strat_id,
                'situation': situation_tokens,  # PMPC
            }
            inputs.append(res)

        if dialog[i]['speaker'] == 'sys':
            text = [strat_id] + text

        context = context + [text]

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    """입력 → Feature 변환"""
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')
    max_situation_length = kwargs.get('max_situation_length', 50)  # PMPC
    
    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
    
    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos,
            ipt['context'], max_input_length,
            ipt['response'], max_decoder_input_length, 
            ipt['strat_id'],
            situation=ipt.get('situation', None),
            max_situation_length=max_situation_length,
        )
        features.append(feat)
    return features


class FeatureDataset(Dataset):
    """PMPC Feature Dataset"""
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeatures], toker: PreTrainedTokenizer, infer=False):
        """배치 collate 함수 - situation 포함"""
        pad = toker.pad_token_id
        if pad is None:
            pad = toker.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = toker.bos_token_id
        if bos is None:
            bos = toker.cls_token_id
            assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = toker.eos_token_id
        if eos is None:
            eos = toker.sep_token_id
            assert eos is not None, 'either eos_token_id or sep_token_id should be provided'
        
        # Input IDs
        input_ids = pad_sequence(
            [torch.tensor(f.input_ids, dtype=torch.long) for f in features],
            batch_first=True, padding_value=pad
        )
        attention_mask = pad_sequence(
            [torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
            batch_first=True, padding_value=0.
        )
        input_length = torch.tensor([f.input_length for f in features], dtype=torch.long)
        
        # Decoder IDs
        if not infer:
            decoder_input_ids = pad_sequence(
                [torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                batch_first=True, padding_value=pad
            )
            labels = pad_sequence(
                [torch.tensor(f.labels, dtype=torch.long) for f in features],
                batch_first=True, padding_value=-100
            )
        else:
            decoder_input_ids = torch.tensor(
                [[f.decoder_input_ids[0]] for f in features], dtype=torch.long
            )
            labels = None
        
        # Strategy ID
        strat_id = torch.tensor(
            [f.labels[0] for f in features], dtype=torch.long
        ) - len(toker) + 8
        
        # PMPC: Situation IDs
        has_situation = any(len(f.situation_ids) > 0 for f in features)
        if has_situation:
            situation_ids = pad_sequence(
                [torch.tensor(f.situation_ids if f.situation_ids else [pad], dtype=torch.long) for f in features],
                batch_first=True, padding_value=pad
            )
            situation_mask = pad_sequence(
                [torch.tensor([1.] * max(f.situation_length, 1), dtype=torch.float) for f in features],
                batch_first=True, padding_value=0.
            )
        else:
            situation_ids = None
            situation_mask = None
        
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'strat_id': strat_id,
        }
        
        # PMPC specific
        if situation_ids is not None:
            res['situation_ids'] = situation_ids
            res['situation_mask'] = situation_mask
        
        return res


class DynamicBatchingLoader(object):
    """Validation data loader"""
    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()

            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                while len(features) >= self.bs:
                    batch = self._batch_feature(features[:self.bs])
                    yield batch
                    features = features[self.bs:]

            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch

        except StopIteration:
            pass
    
    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


def prepare_infer_batch(features, toker, interact=None):
    """Inference batch 준비"""
    res = FeatureDataset.collate(features, toker, True)
    
    res['batch_size'] = res['input_ids'].size(0)

    other_res = res['other_res'] = {}
    other_res['acc_map'] = {
        'cls_strat_id': 'pred_strat_id',
    }

    if interact is None and GOLDEN_TRUTH:
        other_res['cls_strat_id'] = res.get('strat_id')
    else:
        other_res['cls_strat_id'] = res.pop('strat_id')

    return res


def get_infer_batch(infer_input_file, toker, **kwargs):
    """Inference batch 생성"""
    assert 'infer_batch_size' in kwargs, 'you should give infer_batch_size'
    infer_batch_size = kwargs.get('infer_batch_size')

    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()
    
    features = []
    sample_ids = []
    posts = []
    references = []
    for sample_id, line in tqdm.tqdm(enumerate(reader), total=len(reader), desc=f"inferring"):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)
        for i in range(len(inputs)):
            features.append(tmp_features[i])
            ipt = inputs[i]
            posts.append(toker.decode(ipt['context'][-1]))
            references.append(toker.decode(ipt['response']))
            sample_ids.append(sample_id)
    
            if len(sample_ids) == infer_batch_size:
                yield prepare_infer_batch(features, toker), posts, references, sample_ids
                features = []
                sample_ids = []
                posts = []
                references = []

    if len(sample_ids) > 0:
        yield prepare_infer_batch(features, toker), posts, references, sample_ids
