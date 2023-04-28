import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ElemExtractor(nn.Module):
    def __init__(self, model_path, lang, device, grad):
        super(ElemExtractor, self).__init__()
        self.device = device
        self.lang = lang
        self.grad = grad
        self.enc = AutoModel.from_pretrained(model_path)
        self.sent_em = None
        self.sent_pred = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.charge_pred = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.lang.index2charge))
        )

    def forward(self, enc_input, pad_sep_lens):
        enc_input = {k: v.to(self.device) for k, v in enc_input.items()}
        if self.grad == 1:
            outputs = self.enc(**enc_input)['last_hidden_state']
        else:
            with torch.no_grad():
                # [batch_size, seq_len, hidden_dim]
                outputs = self.enc(**enc_input)['last_hidden_state']
        # 得到句子的嵌入表示 [batch_size, sent_count, dim]
        self.sent_em = self.split_tensor(outputs, pad_sep_lens)
        # 预测每个句子的重要性 [batch_size, sent_count, 2]
        pred_score = self.sent_pred(self.sent_em)
        # 根据pred_score选择的句子预测其对应的指控 [batch_size, dim]
        # selected_sents = self.sent_select(pred_score)
        # charge score [batch_size, charge_num]
        # charge_score = self.charge_pred(selected_sents)
        return pred_score.view(pred_score.shape[0]*pred_score.shape[1], -1)

    def split_tensor(self, input, pad_sp_lens):
        max_len = max([len(s) for s in pad_sp_lens])
        splited_tensor = []
        for idx, sp_len in enumerate(pad_sp_lens):
            # [sent_count, dim]
            sample = torch.stack([torch.sum(i, dim=0) for i in torch.split(input[idx], sp_len)], 0)
            splited_tensor.append(F.pad(sample, (0, 0, 0, max_len-sample.shape[0]))) #left,right,top,bottom
        # [batch_size,sent_count, dim]
        return torch.stack(splited_tensor, dim=0)

    def sent_select(self,pred_score):
        """
        :param pred_score: [batch_size, sent_count, 2] 句子重要性预测分数
        :return:
        """
        # [batch_size, snent_cout]
        pred = pred_score.argmax(dim=2).cpu().tolist()
        # [batch_size, dim]
        selected_sents = []
        for b_idx, sent_idxs in enumerate(pred):
            indices = torch.LongTensor([i for i, v in enumerate(sent_idxs) if v==1]).to(self.device)
            selected_sents.append(torch.sum(torch.index_select(self.sent_em[b_idx], 0,indices), dim=0))
        # [batch_size, dim]
        return torch.stack(selected_sents, dim=0)

class CEEE(nn.Module):
    # Charge Enhenced CEEE
    def __init__(self, model_path, lang, device):
        super(CEEE, self).__init__()
        self.device = device
        self.lang = lang
        self.enc = AutoModel.from_pretrained(model_path)
        self.sent_em = None
        self.sent_pred = nn.Sequential(
            nn.Linear(3*768, 768),
            nn.ReLU(),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, enc_fact, pad_sep_lens, enc_desc, dfd_positions):
        enc_fact = {k: v.to(self.device) for k, v in enc_fact.items()}
        enc_desc = {k: v.to(self.device) for k, v in enc_desc.items()}
        enc_fact = self.enc(**enc_fact)['last_hidden_state'] # [batch_size, seq_len, model_dim]
        enc_fact_sent = self.split_tensor(enc_fact, pad_sep_lens)
        enc_desc = self.enc(**enc_desc)['last_hidden_state'] # [charge_num, seq_len, model_dim]
        # token_level_match
        token_matched_enc_fact = self.token_match(enc_fact, enc_desc) # [batch_size, seq_len, model_dim]
        token_matched_enc_fact_sent = self.split_tensor(token_matched_enc_fact, pad_sep_lens)
        # position_info
        position_reps = self.get_dfd_position_rep(enc_fact, pad_sep_lens, dfd_positions)
        position_reps = position_reps.expand(-1, enc_fact_sent.shape[1], -1)
        combine_tensor = torch.concat([enc_fact_sent, token_matched_enc_fact_sent, position_reps], dim=2)
        # 预测每个句子的重要性 [batch_size, sent_count, 2]
        pred_score = self.sent_pred(combine_tensor)
        return pred_score.view(pred_score.shape[0]*pred_score.shape[1], -1)

    def split_tensor(self, input, pad_sp_lens):
        max_len = max([len(s) for s in pad_sp_lens])
        splited_tensor = []
        for idx, sp_len in enumerate(pad_sp_lens):
            # [sent_count, dim]
            sample = torch.stack([torch.mean(i, dim=0) for i in torch.split(input[idx], sp_len)], 0)
            splited_tensor.append(F.pad(sample, (0, 0, 0, max_len-sample.shape[0]))) #left,right,top,bottom
        # [batch_size,sent_count, dim]
        return torch.stack(splited_tensor, dim=0)

    def token_match(self,enc_fact, enc_desc):
        batch_size = enc_fact.shape[0]
        enc_fact = enc_fact.view(-1, enc_fact.shape[2]) # [fact_token_num, model_dim]
        enc_desc = enc_desc.view(-1, enc_desc.shape[2]) # [desc_token_num, model_dim]
        token_match_score = torch.matmul(enc_fact, enc_desc.t()) # [fact_token_num, desc_token_num]
        token_match_score = torch.softmax(token_match_score, dim=1)
        token_matched_enc_fact = torch.matmul(token_match_score, enc_desc) # [fact_token_num, model_dim]
        return token_matched_enc_fact.view(batch_size, -1, enc_fact.shape[1])

    def get_dfd_position_rep(self, fact, pad_sep_lens, dfd_positions):
        sent_em = self.split_tensor(fact, pad_sep_lens)
        positions = []
        for idx, indices in enumerate(dfd_positions):
            sents = torch.index_select(sent_em[idx], 0, torch.tensor(indices).to(self.device))
            sents = sents.t().unsqueeze(dim=0)
            sents =  F.max_pool1d(sents, kernel_size=sents.shape[2],stride=2)
            positions.append(sents.view(-1, 768))
        return torch.stack(positions, dim=0)


class Base(nn.Module):
    def __init__(self, model_path, lang, device):
        super(Base, self).__init__()
        self.device = device
        self.lang = lang
        self.enc = AutoModel.from_pretrained(model_path)

        self.pred = nn.Sequential(
            nn.Linear(768, int(0.5*768)),
            nn.ReLU(),
            nn.Linear(int(0.5*768), len(lang.index2charge))
        )

    def forward(self, enc_input, mask_positions):
        enc_input = {k: v.to(self.device) for k, v in enc_input.items()}
        outputs = self.enc(**enc_input)['last_hidden_state']
        # [sent_count, hidden_dim]
        ts = []
        for idx in range(len(mask_positions)):
            reps = torch.index_select(outputs[idx].squeeze(), 0, torch.tensor(mask_positions[idx]).to(self.device))
            ts.append(reps)
        ts = torch.concat(ts, dim=0)
        outputs = self.pred(ts)
        return outputs

