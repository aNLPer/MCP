import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def set_seed(seed:int):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def label_construct(relevant_sents, sp_lens, pattern):
    labels = []
    max_len = max([len(s) for s in sp_lens])
    for dic in relevant_sents:

        if pattern == "act":
            act_label = [0] * max_len
            for idx in dic['act']:
                act_label[idx] = 1
            labels.extend(act_label)

        if pattern == "sub":
            sub_label = [0] * max_len
            for idx in dic['subjective']:
                sub_label[idx] = 1
            labels.extend(sub_label)

        if pattern == "res":
            res_label = [0] * max_len
            for idx in dic['res']:
                res_label[idx] = 1
            labels.extend(res_label)

        if pattern == "all":
            all_label = [0] * max_len
            temp = []
            for k, v in dic.items():
                temp.extend(v)
            for idx in list(set(temp)):
                all_label[idx] = 1
            labels.extend(all_label)

    return labels

def evaluate(model, data_loader, pattern, fr):
    with torch.no_grad():
        model.eval()
        elem_trues = []
        elem_preds = []
        sent_lens = []
        for ids, inputs, enc_inputs, enc_desc, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, sent_len, pad_sp_lens, relevant_sents, mask_positions, dfd_positions in data_loader:
            # 评估元素抽取
            max_len = max([len(s) for s in pad_sp_lens])
            elem_score = model(enc_inputs, pad_sp_lens, enc_desc, dfd_positions)
            elem_pred = elem_score.argmax(dim=1).cpu().tolist()
            elem_label = Util.label_construct(relevant_sents, pad_sp_lens, pattern)
            elem_trues.extend([ts.tolist() for ts in torch.split(torch.tensor(elem_label), [max_len]*int(len(elem_label)/max_len))])
            elem_preds.extend([ts.tolist() for ts in torch.split(torch.tensor(elem_pred), [max_len] * int(len(elem_label)/max_len))])
            sent_lens.extend(sent_len)
        elem_metric(y_preds=elem_preds, y_trues=elem_trues, sent_lens=sent_lens,fr=fr)

def elem_metric(y_preds, y_trues, sent_lens, fr):
    """
    Extraction Match at sent level and token level
    :param y_preds:
    :param y_trues:
    :param sp_lens:
    :return:
    """
    case_count = 0
    sent_score = 0.
    token_score = 0.
    for pred, truth, sent_len in zip(y_preds, y_trues, sent_lens): # 便利每个例子
        if truth == [0]*len(truth): # 标签全为0，即无效数据
            continue
        else:# 有效数据
            case_count += 1
            if pred == truth:
                sent_score += 1.0
                token_score += 1.0
                continue
            p_set = set([idx for idx, val in enumerate(pred) if val == 1])
            t_set = set([idx for idx, val in enumerate(truth) if val == 1])
            matched_sent_idx = list(p_set.intersection(t_set))
            if len(matched_sent_idx) == 0:
                continue
            # sentence level
            sent_score += 2*len(matched_sent_idx)/(len(t_set)+len(p_set)+0.0001)
            # token leval
            matched_token_count = 0
            totall_token_count = 0
            for sent_idx in t_set:
                totall_token_count += sent_len[sent_idx]
                if sent_idx in matched_sent_idx:
                    matched_token_count += sent_len[sent_idx]
            for sent_idx in p_set:
                if sent_idx >= len(sent_len):
                    continue
                totall_token_count += sent_len[sent_idx]
                if sent_idx in matched_sent_idx:
                    matched_token_count += sent_len[sent_idx]
            token_score += matched_token_count/(totall_token_count+0.0001)
    print(f"Elem: sent_match: {round(sent_score/case_count, 2)}   'token_match: {round(token_score/case_count,2)}\n")
    fr.write(f"Elem: sent_match: {round(sent_score/case_count, 2)}   'token_match: {round(token_score/case_count,2)}\n")
    fr.flush()

def charge_metric(y_preds, y_trues, right_case, total_case, fr):
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    precision = precision_score(y_true=y_trues, y_pred=y_preds, average="macro")
    recall = recall_score(y_true=y_trues, y_pred=y_preds, average="macro")
    f1 = f1_score(y_true=y_trues, y_pred=y_preds, average="macro")
    fr.write(classification_report(y_true=y_trues, y_pred=y_preds))
    print(f"Charge: acc: {round(acc, 2)}   "
          f"precision: {round(precision, 2)}   "
          f"recall: {round(recall, 2)}   "
          f"f1: {round(f1, 2)}   "
          f"acc_: {round(right_case / total_case, 2)}")
    fr.write(f"acc: {round(acc, 2)}   "
             f"precision: {round(precision, 2)}   "
             f"recall: {round(recall, 2)}   "
             f"f1: {round(f1, 2)}   "
             f"acc_: {round(right_case / total_case, 2)}\n")
    fr.flush()