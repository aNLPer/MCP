import os,json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import util
from preprocess import Processor
from model import ElemExtractor, CEEE
import argparse
import setting
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from dataset import ClozeDataset, Lang
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(seed, enc_name, enc_path, data_path, params, charge_desc):
    print("preparing dataset...")
    train_path, dev_path, test_path = f"./datasets/{data_path}/train.json",f"./datasets/{data_path}/dev.json",f"./datasets/{data_path}/test.json"
    lang = Lang(train_path)
    # 去除数据集中不存在得charge
    d = list(set(charge_desc.keys()).difference(set(lang.index2charge)))
    if len(d)>0:
        charge_desc.pop(d[0])
    processor = Processor(lang)
    train_data, dev_data, test_data = processor.load_data(train_path), processor.load_data(dev_path), processor.load_data(test_path)
    dataset_train = ClozeDataset(train_data, enc_path, lang, charge_desc)
    dataset_dev = ClozeDataset(dev_data, enc_path, lang, charge_desc)
    dataset_test = ClozeDataset(test_data, enc_path, lang, charge_desc)
    train_data_loader = DataLoader(dataset_train, batch_size=params['batch_size'], collate_fn=dataset_train.collate_fn, shuffle=True)
    dev_data_loader = DataLoader(dataset_dev, batch_size=params['batch_size'], collate_fn=dataset_dev.collate_fn, shuffle=False)
    test_data_loader = DataLoader(dataset_test, batch_size=params['batch_size'], collate_fn=dataset_test.collate_fn, shuffle=False)

    print("creating model...")
    model = CEEE(enc_path, lang, device)
    model.to(device)
    # 定义损失函数，优化器，学习率调整器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=params['lr'])
    warmup_step = int(0.1*(len(train_data)/params["batch_size"])*params['epoch']+1)
    training_step = int((len(train_data)/params["batch_size"])*params['epoch']+1)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=warmup_step,
                                                                   num_training_steps=training_step,
                                                                   num_cycles=1)
    print("training model...")
    from tqdm import tqdm
    report_file = open(f"./outputs/reports/{enc_name}_{seed}_{data_path}.txt", "w", encoding="utf-8")
    for e in range(params['epoch']):
        print(f"-------------------------------epoch:{e + 1}-------------------------------------")
        model.train()
        train_loss = 0
        for ids, inputs, enc_inputs, enc_desc, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, sent_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions in tqdm(train_data_loader):
            # 梯度置零
            optimizer.zero_grad()
            sent_scores = model(enc_inputs, pad_sp_lens, enc_desc, dfd_positions)
            labels = util.label_construct(relevant_sents, pad_sp_lens, params["pattern"][0])
            loss = criterion(sent_scores, torch.tensor(labels).to(device))
            train_loss+=loss.item()
            # 累计梯度
            loss.backward()
            # 梯度
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新梯度
            optimizer.step()
            # 更新学习率
            scheduler.step()
        print(f"train_loss:{round(train_loss / len(train_data_loader.dataset), 4)}")
        prefix = f"{enc_name}_{seed}_{data_path}_{e}"
        torch.save(model, f"./outputs/models/{prefix}.pkl")
        util.evaluate(model, dev_data_loader, params['pattern'][0], report_file)
        util.evaluate(model, test_data_loader, params['pattern'][0], report_file)

def main():
    print("Running ...")
    params = setting.params
    encs = setting.encs
    charge_desc = setting.charge_desc
    for seed in params["seeds"][:1]:
        print(f"set seed {seed}")
        util.set_seed(seed)
        for enc_name, enc_path in encs.items():
            for pattern in params["pattern"]:
                for data_path in params["data_path"]:
                    print(f"seed: {seed}\n"
                          f"enc_name: {enc_name}\n"
                          f"pattern: {pattern}\n"
                          f"data: {data_path}\n"
                          f"batch_size: {params['batch_size']}\n"
                          f"lr: {params['lr']}\n")
                    train(seed, enc_name, enc_path, data_path, params, charge_desc)
if __name__=="__main__":
    main()