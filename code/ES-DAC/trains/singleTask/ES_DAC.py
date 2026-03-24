import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import MetricsTop, dict_to_str


logger = logging.getLogger('MSA')




class ES_DAC():
    def __init__(self, args):
        self.args = args
        self.MSEloss = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.log_alpha = torch.nn.Parameter(torch.zeros(4))  # [M, T, A, V]

    
    def do_train(self, model, dataloader):

        optimizer = optim.Adam([
            {'params': [self.log_alpha], 'lr': self.args.log_lr},
            {'params': model.Model.text_bert_encoder.parameters(), 'lr': self.args.bert_text_lr},
            {'params': model.Model.audio_trans.parameters(), 'lr': self.args.bert_audio_lr},
            {'params': model.Model.video_trans.parameters(), 'lr': self.args.bert_video_lr},
            {'params': model.Model.audio_LLD_block.a_MFCC_encoder.parameters(), 'lr': self.args.MFCC_lr},
            {'params': model.Model.audio_LLD_block.a_SMA_encoder.parameters(), 'lr': self.args.SMA_lr},
            {'params': [p for name, p in model.named_parameters() 
                       if not any(x in name for x in ['text_bert_encoder', 'audio_trans', 'video_trans', 
                                                     'audio_LLD_block.a_MFCC_encoder', 'audio_LLD_block.a_SMA_encoder'])], 
             'weight_decay': 1e-5, 'lr': self.args.learning_rate},
        ], lr=self.args.learning_rate
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',  
            factor=0.5,  
            patience=self.args.patience,
            verbose=True
        )

        epochs, best_epoch = 0, 0

        epoch_results = {
            'train':[],
            'valid':[],
            'test':[],
        }

        min_or_max = 'min' if self.args.KeyEval == 'Loss' else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = model
        net_q = model
        net_q.load_state_dict(net.state_dict())

        for epochs in range(0, self.args.epochs):
            epochs += 1
            y_pred, y_true = [], []
            net.train()

            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    audio_LLD = batch_data['audio_LLD'].to(self.args.device)

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    
                    optimizer.zero_grad()
                    OutPuts = model(text, audio, vision, audio_LLD)
                    outputs = OutPuts['M']
                    out_T = OutPuts['T']
                    out_A = OutPuts['A']
                    out_V = OutPuts['V']
                    

                    weights = torch.softmax(self.log_alpha.clamp(-10, 10), dim=0)

                    alpha_M, alpha_T, alpha_A, alpha_V = weights

                    # compute loss
                    loss_M = self.MSEloss(outputs, labels)
                    loss_T = self.MSEloss(out_T, labels)
                    loss_A = self.MSEloss(out_A, labels)
                    loss_V = self.MSEloss(out_V, labels)

                    loss = alpha_M * loss_M + alpha_T * loss_T + alpha_A * loss_A + alpha_V * loss_V

                    # backward
                    # update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            logger.info(
                f">>Epoch:{epochs}:"
                f"Train-({self.args.model_name}) [{epochs - best_epoch}/epochs:{epochs}/seed:{self.args.cur_seed}]\n"
                f"alpha_M: {alpha_M.item():.8f} "
                f"alpha_T: {alpha_T.item():.8f} "
                f"alpha_A: {alpha_A.item():.8f} "
                f"alpha_V: {alpha_V.item():.8f} "
                f"Loss: {train_loss:.8f} \n"
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(net, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]

            scheduler.step(cur_valid)
            
            if epochs == self.args.epochs:
                save_path = (self.args.model_save_dir / str(self.args.dataset_name) / f"{self.args.model_name}{self.args.result_name}_{self.args.dataset_name}_{epochs}.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)

            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            train_results["Loss"] = train_loss
            epoch_results['train'].append(train_results)
            epoch_results['valid'].append(val_results)
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            epoch_results['test'].append(test_results)


        return epoch_results

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    audio_LLD = batch_data['audio_LLD'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(text, audio, vision, audio_LLD)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        sample_results.extend(np.atleast_1d(preds.squeeze()))
                    
                    loss =  self.MSEloss(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 8)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
