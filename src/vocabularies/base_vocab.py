__author__ = 'Connor Heaton'

import os
import json

from ..utils import jsonKeys2int, try_cast_int


class BaseVocab(object):
    def __init__(self):
        # super(BaseVocab, self).__init__()
        self.vocab = {}
        self.reverse_vocab = {}

        self.key_component_names = []
        self.sep_char = '|'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'

    def __len__(self):
        if self.vocab is None:
            return 0
        else:
            return len(self.vocab)

    def read_vocab(self, vocab_dir):
        print('Reading vocab...')
        vocab_fp = os.path.join(vocab_dir, 'vocab.json')
        reverse_vocab_fp = os.path.join(vocab_dir, 'reverse_vocab.json')

        self.vocab = json.load(open(vocab_fp))
        self.reverse_vocab = json.load(open(reverse_vocab_fp), object_hook=jsonKeys2int)
        print('self.reverse_vocab[0]: {}'.format(self.reverse_vocab[0]))

    def get_item(self, item_id):
        item = self.reverse_vocab.get(item_id, None)
        return item

    def get_item_components(self, item_id):
        # print('item_id: {}'.format(item_id))
        item_key = self.get_item(item_id)
        # print('item_id: {} item_key: {}'.format(item_id, item_key))
        # print('item_id: {} item_key: {}'.format(item_id, item_key))
        if item_key in [self.bos_token, self.eos_token]:
            item_components = [item_key for _ in self.key_component_names]
        elif item_key is None:
            item_components = ['B/S' for _ in self.key_component_names]
        else:
            item_components = [try_cast_int(v) for v in item_key.split(self.sep_char)]
        # print('item_components: {}'.format(item_components))
        # print('key_component_names: {}'.format(self.key_component_names))

        component_dict = dict(zip(self.key_component_names, item_components))
        return component_dict

