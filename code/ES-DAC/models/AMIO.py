"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .singleTask import *
from .subNets import AlignSubNet
from .singleTask.ES_DAC import ES_DAC

# from transformers import BertConfig

from transformers import BertModel, BertConfig

# config = BertConfig.from_pretrained(r'C:\Users\sixinheyi\.cache\huggingface\transformers\bert-base-uncased-local')
# model = BertModel.from_pretrained(r'C:\Users\sixinheyi\.cache\huggingface\transformers\bert-base-uncased-local', config=config)

# tokenizer = BertTokenizer.from_pretrained(r"/public/home/sixinheyi/.cache/huggingface/transformers/bert-base-uncased-local")
# model = BertModel.from_pretrained(r"/public/home/sixinheyi/.cache/huggingface/transformers/bert-base-uncased-local")

import os
# # 定义两个路径
# path1 = r'C:\Users\sixinheyi\.cache\huggingface\transformers\bert-base-uncased-local'
# path2 = r"/public/home/sixinheyi/.cache/huggingface/transformers/bert-base-uncased-local"

# bert_path = path1 if os.path.exists(path1) else path2

# config = BertConfig.from_pretrained(bert_path)
# model = BertModel.from_pretrained(bert_path)

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            # single-task
            'es_dac': ES_DAC,
        }
        self.need_model_aligned = args.get('need_model_aligned', None)
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if(self.need_model_aligned):
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()
        lastModel = self.MODEL_MAP[args['model_name']]

        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, audio_x_LLD, *args, **kwargs):
        if(self.need_model_aligned):
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x, audio_x_LLD,  *args, **kwargs)
