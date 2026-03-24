import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import os
from pathlib import Path

# # 定义两个路径
# path1 = r'C:\Users\sixinheyi\.cache\huggingface\transformers\bert-base-uncased-local'
# path2 = r"/public/home/sixinheyi/.cache/huggingface/transformers/bert-base-uncased-local"

# bert_path = path1 if os.path.exists(path1) else path2

# tokenizer = BertTokenizer.from_pretrained(bert_path)
# model = BertModel.from_pretrained(bert_path)

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}

class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert-base-uncased', pretrained='bert-base-uncased'):
        super().__init__()

        # tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        # model_class = TRANSFORMERS_MAP[transformers][0]
        # self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        # self.model = model_class.from_pretrained(pretrained)
        # self.use_finetune = use_finetune

        # 指定本地BERT模型路径
        model_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--bert-base-uncased' / 'snapshots' / 'main'

        print(f"BertTextEncoder - pretrained: {pretrained}")
        
        # 检查传入的pretrained路径是否存在
        if os.path.exists(pretrained):
            print(f"Use local paths: {pretrained}")
            model_path = pretrained
        else:
            print(f"The local path does not exist: {pretrained}")
            print("The default path will be used: ")
            model_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--bert-base-uncased' / 'snapshots' / 'main'
            print(f"Use local paths: {model_path}")
        
        # 使用本地模型
        self.tokenizer = BertTokenizer.from_pretrained(str(model_path))
        self.model = BertModel.from_pretrained(str(model_path))
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    # def from_text(self, text):
    #     """
    #     text: raw data
    #     """
    #     input_ids = self.get_id(text)
    #     with torch.no_grad():
    #         last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
    #     return last_hidden_states.squeeze()
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
