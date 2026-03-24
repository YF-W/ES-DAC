import torch
import torch.nn as nn
import torch.nn.functional as F
from ..subNets.transformers_encoder.transformer import TransformerEncoder
from ..subNets import BertTextEncoder, SubNet, TextSubNet


class TextSubNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

class ES_DAC(nn.Module):
    def __init__(self, args):
        super(ES_DAC, self).__init__()

        # define
        self.text_in, self.audio_in, self.video_in, self.a_LLD_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_prob, self.audio_prob, self.video_prob, self.a_LLD_prob = args.dropouts
        self.text_dim, self.audio_dim, self.video_dim, self.LLD_dim = args.feature_dims
        self.text_len = self.audio_len = self.video_len = args.feature_lens[0]
        self.text_layer, self.audio_layer, self.video_layer = args.feature_layers
        self.attn_dropout_t, self.attn_dropout_a, self.attn_dropout_v, self.attn_dropout_a_LLD = args.dropouts
        self.num_head_t, self.num_head_a, self.num_head_v, self.num_head_m = args.num_heads
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout 
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        self.output_dim = 1
        self.all_seq_len = args.all_seq_len

        self.text_sub_out = args.text_sub_out
        self.text_bert_out = args.text_bert_out

        self.audio_hidden_out = args.audio_hidden_out

        self.video_conv = (self.video_in //5 ) * 5
        self.video_trans_out = args.video_trans_out

        self.dataset_name = args.dataset_name
        # text
        if self.dataset_name in ('mosei', 'mosi'):
            # text subnets
            self.text_bert_encoder = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)
        elif self.dataset_name in ('sims', 'simsv2'):
            self.text_bert_encoder = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)
        
        self.text_conv1d_en = nn.Conv1d(self.text_len, self.all_seq_len, kernel_size=5, stride=1, padding=2)
        self.text_bert_layer = nn.Linear(self.text_in, self.text_bert_out)


        self.text_out = nn.Linear(self.text_bert_out, self.output_dim)

        # video
        self.video_conv1d_1 = nn.Conv1d(self.video_in, self.video_conv, kernel_size=5, stride=1, padding=2)
        self.video_trans = self.get_network(self_type='v', layers=self.video_layer)
        self.video_conv1d_en = nn.Conv1d(self.video_len, self.all_seq_len, kernel_size=5, stride=1, padding=2)
        self.vidio_trans_layer = nn.Linear(self.video_conv, self.video_trans_out)

        self.video_out = nn.Linear(self.video_trans_out, self.output_dim)

        # audio
        self.audio_lstm = self.get_lstm(self_type='audio')
        self.audio_trans = self.get_network(self_type='a', layers=self.audio_layer)
        self.trans_layer = nn.Linear(self.audio_in, self.audio_hidden_out)
        self.audio_conv1d_en = nn.Conv1d(self.audio_len, self.all_seq_len, kernel_size=5, stride=1, padding=2)
        self.audio_dropout = nn.Dropout(self.audio_prob)

        self.audio_out = nn.Linear(self.audio_hidden_out, self.output_dim)

        self.audio_LLD_block = audio_LLD_block(args)
        # m
        self.m_out = nn.Linear(self.text_bert_out, self.output_dim)


        self.Gavgplool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.Tahn = nn.Tanh()


    def get_network(self, self_type='t', layers=-1):
        if self_type in ['t']:
            embed_dim, attn_dropout, num_heads, selflayers = self.d_t, self.attn_dropout_t, self.num_head_t, self.text_layer
        elif self_type in ['a']:
            embed_dim, attn_dropout, num_heads, selflayers  = self.audio_in, self.attn_dropout_a, self.num_head_a, self.audio_layer
        elif self_type in ['v',]:
            embed_dim, attn_dropout, num_heads, selflayers = self.video_conv, self.attn_dropout_v, self.num_head_v, self.video_layer
        elif self_type in ['a_LLD', 'a_F0', 'a_MFCC', 'a_SMA', 'a_Loudness']:
            embed_dim, attn_dropout,num_heads, selflayers = self.d_a_LLD * 3, self.attn_dropout_a_LLD, self.num_head_a, self.audio_layer
        elif self_type in ['m_t', 'm_a', 'm_v']:
            embed_dim, attn_dropout, num_heads, selflayers = self.d_t, self.attn_dropout, self.num_head_m, self.text_layer
        elif self_type == "m":
            embed_dim, attn_dropout, num_heads, selflayers = self.d_m, self.attn_dropout, self.num_head_m, self.text_layer
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    layers=max(selflayers, layers),
                                    attn_dropout=attn_dropout,
                                    relu_dropout=self.relu_dropout,
                                    res_dropout=self.res_dropout,
                                    embed_dropout=self.embed_dropout,
                                    attn_mask=self.attn_mask)
    
    def get_lstm(self, self_type = 'audio'):
        if self_type in ['audio']:
            audio_in, audio_hidden_out = self.audio_in, self.audio_hidden_out
        elif self_type in ['LLD']:
            audio_in, audio_hidden_out = self.LLD_dim, self.LLD_dim

        else:
            raise ValueError("Unknown network type")


        return nn.LSTM(
                    input_size=audio_in,
                    hidden_size=audio_hidden_out,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False
                    )

    def forward(self, text, audio, video, audio_LLD):
        
        LLD_res = self.audio_LLD_block(audio_LLD)
        # F0 = LLD_res['F0']
        # MFCC = LLD_res['MFCC']
        # SMA = LLD_res['SMA']
        # Loudness = LLD_res['Loudness']
        Spectral_to_text = LLD_res['Spectral_to_text']
        Atten_Prosody = LLD_res['Atten_Prosody']
        LLD_M = LLD_res['LLD_M']
        LLD_M_result = LLD_res['LLD_M_result']

        x_text_1 = self.text_bert_encoder(text)
        x_text_1 = x_text_1 + Spectral_to_text
        
        x_text_1 = self.text_conv1d_en(x_text_1)
        x_text_1 = self.text_bert_layer(x_text_1)

        x_audio_1 = self.audio_trans(audio)
        x_audio_1 = self.trans_layer(x_audio_1)
        x_audio_1 = self.audio_conv1d_en(x_audio_1)

        video = video.transpose(1, 2)
        video = self.video_conv1d_1(video)
        video = video.transpose(1, 2)
        x_video_1 = self.video_trans(video)
        x_video_1 = self.video_conv1d_en(x_video_1)
        x_video_1 = self.vidio_trans_layer(x_video_1)

        m_tav = self.Tahn(x_text_1 + x_audio_1 + x_video_1)
        m_tav = m_tav * Atten_Prosody
        m_ta = self.Tahn(x_text_1 + x_audio_1)
        m_tv = self.Tahn(x_text_1 + x_video_1)
        m_av = self.Tahn(x_audio_1 + x_video_1)

        text_out = 0.25 * m_ta + 0.25 * m_tv + 0.5 * x_text_1
        audio_out = 0.25 * m_ta + 0.25 * m_av + 0.5 * x_audio_1
        video_out = 0.25 * m_tv + 0.25 * m_av + 0.5 * x_video_1

        m_out = m_tav * (self.sigmoid(m_ta + m_tv + m_av))
        m_out = m_out * (1 + self.sigmoid(LLD_M))

        text_out = self.Gavgplool(self.text_out(text_out)[:, :, 0])
        audio_out = self.Gavgplool(self.audio_out(audio_out)[:, :, 0])
        video_out = self.Gavgplool(self.video_out(video_out)[:, :, 0])
        m_out = self.Gavgplool(self.m_out(m_out)[:, :, 0])

        res = {
            'T': text_out,
            'A': audio_out,
            'V': video_out,
            'A_LLD': LLD_M_result,
            "M": m_out
        }

        return res
    

class audio_LLD_block(nn.Module):
    def __init__(self, args):
        super(audio_LLD_block, self).__init__()

        # define 
        self.F0_size = 2
        self.MFCC_size = 4
        self.SMA_size = 6
        self.Loudness_size = 1
        self.F0_hidden_out, self.MFCC_hidden_out, self.SMA_hidden_out, self.Loudness_hidden_out = args.LLDs_hidden_out
        self.len = args.LLDs_len
        self.LLD_dim = args.LLD_dim
        self.batch_size = args.batch_size
        self.text_len = args.feature_lens[0]
        self.text_dim = args.feature_dims[0]
        self.all_seq_len = args.all_seq_len
        self.target_len = args.target_len
        self.out_dim = 1

        self.StrengthChangeDetectionModule_F0 = StrengthChangeDetectionModule(target_len=self.target_len)
        self.StrengthChangeDetectionModule_Loudness = StrengthChangeDetectionModule(target_len=self.target_len)

        self.a_F0_encoder = self.get_lstm('F0')
        self.a_MFCC_encoder = self.get_lstm('MFCC')
        self.a_SMA_encoder = self.get_lstm('SMA')
        self.a_Loudness_encoder = self.get_lstm('Loudness')

        self.a_F0_layer_1 = nn.Linear(self.len, self.LLD_dim)
        self.a_MFCC_layer_1 = nn.Linear(self.len, self.LLD_dim)
        self.a_SMA_layer_1 = nn.Linear(self.len, self.LLD_dim)
        self.a_Loudness_layer_1 = nn.Linear(self.len, self.LLD_dim)

        self.SpatialAttention = SpatialAttention(1, self.LLD_dim)
        self.Atten_Prosody_conv1 = nn.Conv1d(1, self.all_seq_len, kernel_size=1)

        self.SpatialAttention_LLD = SpatialAttention(1, self.LLD_dim)

        self.Spectral_to_text_layer = nn.Linear(self.LLD_dim * self.LLD_dim, self.text_dim)
            
        self.Spectral_to_text_conv1 = nn.Conv1d(1, self.text_len, kernel_size=1)

        self.LLD_M_result_layer = nn.Linear(self.LLD_dim, self.out_dim)
        self.LLD_out_conv1d_1 = nn.Conv1d(self.LLD_dim, self.all_seq_len, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.Gavgplool = nn.AdaptiveAvgPool1d(1)


    def get_lstm(self, self_type = 'audio'):
        if self_type in ['F0']:
            audio_in, audio_hidden_out = self.F0_size, self.F0_hidden_out
        elif self_type in ['MFCC']:
            audio_in, audio_hidden_out = self.MFCC_size, self.MFCC_hidden_out
        elif self_type in ['SMA']:
            audio_in, audio_hidden_out = self.SMA_size, self.SMA_hidden_out
        elif self_type in ['Loudness']:
            audio_in, audio_hidden_out = self.Loudness_size, self.Loudness_hidden_out
        else:
            raise ValueError("Unknown network type")

        return nn.LSTM(
                    input_size=audio_in,
                    hidden_size=audio_hidden_out,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False
                    )

    def forward(self, audio_LLD):
        
        batch_size = audio_LLD.size(0)

        a_F0 = audio_LLD[:, :, 0:2]
        a_F01 = audio_LLD[:, :, 0:1].squeeze(-1)
        a_F02 = audio_LLD[:, :, 1:2].squeeze(-1)
        a_MFCC = audio_LLD[:, :, 10:14]
        a_SMA = audio_LLD[:, :, 18:24]
        a_Loudness = audio_LLD[:, :, -1:].squeeze(-1)
        a_Loudness_lstm = audio_LLD[:, :, -1:]


        a_F01 = self.StrengthChangeDetectionModule_F0(a_F01)
        a_F02 = self.StrengthChangeDetectionModule_F0(a_F02)
        a_F0_1 = torch.einsum('bi,bj->bij', a_F01, a_F02)

        a_Loudness_1 = self.StrengthChangeDetectionModule_Loudness(a_Loudness)
        a_Loudness_1 = torch.einsum('bi,bj->bij', a_Loudness_1, a_Loudness_1)
        a_Loudness_1  = self.sigmoid(a_Loudness_1)

        a_F0_Loudness = a_F0_1 * a_Loudness_1
 
        a_F0_lstm_1, _ = self.a_F0_encoder(a_F0)
        a_MFCC_1, _ = self.a_MFCC_encoder(a_MFCC)
        a_SMA_1, _ = self.a_SMA_encoder(a_SMA)
        a_Loudness_lstm_1, _ = self.a_Loudness_encoder(a_Loudness_lstm)

        a_F0_2 = a_F0_lstm_1.squeeze(-1)
        a_MFCC_2 = a_MFCC_1.squeeze(-1)
        a_SMA_2 = a_SMA_1.squeeze(-1)
        a_Loudness_2 = a_Loudness_lstm_1.squeeze(-1)

        a_F0_3 = self.a_F0_layer_1(a_F0_2)
        a_MFCC_3 = self.a_MFCC_layer_1(a_MFCC_2)
        a_SMA_3 = self.a_SMA_layer_1(a_SMA_2)
        a_Loudness_3 = self.a_Loudness_layer_1(a_Loudness_2)

        a_F0_Loudness_lstm = torch.einsum('bi,bj->bij', a_F0_3, a_Loudness_3)
        a_F0_Loudness_to_M = a_F0_Loudness * self.sigmoid(a_F0_Loudness_lstm) + a_F0_Loudness_lstm * self.sigmoid(a_F0_Loudness)

        a_MFCC_SMA = torch.einsum('bi,bj->bij', a_MFCC_3, a_SMA_3)
        a_LLD_M = a_F0_Loudness_to_M * self.sigmoid(a_MFCC_SMA) + a_MFCC_SMA * self.sigmoid(a_F0_Loudness_to_M)
        a_LLD_M_out = self.LLD_out_conv1d_1(a_LLD_M)
        a_LLD_M_SA = self.SpatialAttention_LLD(a_LLD_M, self.LLD_dim)
        a_LLD_M_SA_1 = torch.einsum('bi,bj->bij', a_LLD_M_SA, a_LLD_M_SA)
        a_LLD_M_SA_1 = a_LLD_M * self.sigmoid(a_LLD_M_SA_1) + a_LLD_M_SA_1 * self.sigmoid(a_LLD_M)
        a_LLD_M_result = self.Gavgplool(a_LLD_M_SA_1).squeeze(-1)
        a_LLD_M_result = self.LLD_M_result_layer(a_LLD_M_result)

        Atten_Prosody = self.SpatialAttention(a_F0_Loudness, self.LLD_dim)
        Atten_Prosody = Atten_Prosody.unsqueeze(1)
        Atten_Prosody = self.Atten_Prosody_conv1(Atten_Prosody)

        a_MFCC_SMA_1 = a_MFCC_SMA.view(batch_size, -1)

        Spectral_to_text = self.Spectral_to_text_layer(a_MFCC_SMA_1)
        Spectral_to_text = Spectral_to_text.unsqueeze(1)
        Spectral_to_text = self.Spectral_to_text_conv1(Spectral_to_text)

        res = {
            'F0': a_F0_3,
            # 'F0': a_F0,
            'MFCC': a_MFCC_3,
            'SMA': a_SMA_3,
            'Loudness': a_Loudness_3,
            # 'Loudness': a_Loudness,
            'Spectral_to_text': Spectral_to_text,
            'Atten_Prosody': Atten_Prosody,
            'LLD_M': a_LLD_M_out,
            'LLD_M_result': a_LLD_M_result
        }

        return res


class SpatialAttention(nn.Module):
    def __init__(self, in_channels=1, LLD_dim=64):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_map_layer = nn.Linear(LLD_dim * LLD_dim, LLD_dim)


    def forward(self, x, LLD_dim):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch_size = x.size(0)

        attention_map = self.conv1(x)
        attention_map = attention_map.view(x.size(0), -1)
        attention_map = self.softmax(attention_map)
        attention_map = attention_map.view(x.size(0), 1, LLD_dim, LLD_dim)
        
        attention_map_1 = attention_map * x 

        attention_map_2 = attention_map_1.view(batch_size, -1)

        out = self.attention_map_layer(attention_map_2)

        return out

class StrengthChangeDetectionModule(nn.Module):
    def __init__(self, target_len=256):
        super(StrengthChangeDetectionModule, self).__init__()

        self.target_len = target_len

    def forward(self, x):

        batch_size, seq_len = x.shape
        segment_size = seq_len / self.target_len 

        output = []
        for i in range(self.target_len):
            start = int(round(i * segment_size))
            end = int(round((i + 1) * segment_size))
            if end > seq_len: 
                end = seq_len
            segment = x[:, start:end]

            if segment.shape[1] <= 1:
                change_strength = torch.zeros(batch_size, device=x.device)
            else: 
                diff = torch.diff(segment, dim=1).abs()
                change_strength = diff.mean(dim=1)

            output.append(change_strength.unsqueeze(1))

        return torch.cat(output, dim=1)



  