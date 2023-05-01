import torch
import json
from model import Base, BaseWP
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

def evaluate(lang, model, epoch, data_loader, report_file, prefix, mode):
    logits = []  # 记录logits
    with torch.no_grad():
        model.eval()
        y_trues = []
        y_preds = []
        for ids, inputs, enc_inputs, enc_desc, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, sent_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions in data_loader:
            if isinstance(model, Base):
                pred_score = model(enc_inputs, mask_positions) #enc_desc=enc_desc,pad_sp_lens=pad_sp_lens,
            if isinstance(model, BaseWP):
                pred_score = model(enc_inputs, mask_positions, pad_sp_lens, dfd_positions)
            pred = pred_score.argmax(dim=1).cpu().tolist()
            y_preds.extend(pred)
            y_trues.extend(charge_idxs)
            logits.extend(list(zip(ids, inputs, pred_score.cpu().tolist(), pred, [lang.index2charge[p] for p in pred], charge_idxs, [lang.index2charge[l] for l in charge_idxs])))
        acc_ = acc_case_level(logits)
        acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
        precision = precision_score(y_true=y_trues, y_pred=y_preds, average="macro")
        recall = recall_score(y_true=y_trues, y_pred=y_preds, average="macro")
        f1 = f1_score(y_true=y_trues, y_pred=y_preds, average="macro")
        report_file.write(f"--------------------------------{epoch}-------------------------------\n")
        report_file.write(classification_report(y_true=y_trues, y_pred=y_preds))
        report_file.write(f"acc: {round(acc, 2)}   precision: {round(precision, 2)} recall: {round(recall, 2)}   f1: {round(f1, 2)}, acc_: {acc_}\n")
        print(f"acc: {round(acc, 2)}   precision: {round(precision, 2)} recall: {round(recall, 2)}   f1: {round(f1, 2)}, acc_: {acc_}")
        with open(f"./outputs/logits/{prefix}_{epoch}_{mode}.txt", "w", encoding="utf-8") as fo:
            for item in logits:
                fo.write(json.dumps(item, ensure_ascii=False)+"\n")

def acc_case_level(logits):
    ids = [logit[0] for logit in logits]
    sp_lens = []
    i = 0
    j = i+1
    while i<len(ids) and j<len(ids):
        while ids[i] == ids[j] and j+1<len(ids):
            j += 1
        sp_lens.append(j-i)
        i = j
        j = j+1
    sp_lens[-1]+=1
    # preds
    preds = [ts.tolist() for ts in torch.split(torch.tensor([logit[3] for logit in logits]), sp_lens, dim=0)]
    # trues
    trues = [ts.tolist() for ts in torch.split(torch.tensor([logit[5] for logit in logits]), sp_lens, dim=0)]

    right_case = 0
    case_num = 0
    for pred, true in zip(preds, trues):
        case_num +=1
        if pred == true:
            right_case+=1
    return round(right_case/case_num,2)


