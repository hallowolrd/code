import collections
import math
import torch
from args import parse
import argparse
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch import nn
from utils import *

class Client:
    def __init__(self,parse:argparse.ArgumentParser,client_id:int,logger,c_T:int):
        self.args = parse.parse_args()
        self.client_id = client_id
        self.model = self.load_client_model()
        self.device = self.args.device
        self.c_T =  c_T
        self.client_epochs = self.args.client_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.batch_size = self.args.batch_size
        self.train_loader,self.val_loader,self.test_loader = None,None,None
        self.get_dataloader()

        self.logger = logger
        self.upload_expert_id = 0
        self.acc_list = []
        self.optimizer_list = []
        for expert in self.model.moe.experts:
            self.optimizer_list.append(optim.Adam(expert.parameters(), lr=self.args.learning_rate))

        self.kl_loss = KLDivergenceLoss()

    def load_client_model(self):
        self.model_path = self.args.model_save_path + f"/{self.client_id}.pth"
        return torch.load(self.model_path,weights_only=False)

    def save_client_model(self):
        torch.save(self.model, self.model_path)

    def load_client_data(self):
        dataset_list = []
        for str in ["train","val","test"]:
            data_path = self.args.data_save_path + f"/{self.client_id}_{str}.pth"
            dataset_list.append(torch.load(data_path))
        return dataset_list

    def get_dataloader(self):
        train_dataset,val_dataset,test_dataset = self.load_client_data()
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


    def renew_model(self):
        self.model.to(self.device)
        global_model = torch.load(self.args.model_save_path + f"/server.pth", weights_only=False).to(self.device)
        global_parm = global_model.experts["SubMoELayer"]
        self.model.moe.experts[-1] = global_parm
        self.model.conv.load_state_dict(global_model.conv.state_dict())



    def evaluate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs,_,_,_ = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        return val_loss, val_acc

    def train(self):
        self.renew_model()

        self.model.to(self.device)

        # global expert --kd--> other experts
        for inputs, _ in self.val_loader:
            inputs= inputs.to(self.device)
            _,feature,_,_ = self.model(inputs)
            g_rep = self.model.moe.experts[-1](feature)
            for i,expert in enumerate(self.model.moe.experts[:-1]):
                kl = self.kl_loss(g_rep,expert(feature))
                torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1)
                self.optimizer_list[i].zero_grad()
                kl.backward(retain_graph=True)
                self.optimizer_list[i].step()
        self.model.moe.reset_activation_counts()
        self.model.moe.experts[-1].reset_activation_counts()

        # normal train process
        for epoch in range(self.client_epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.zero_grad()
                outputs,_,act,shared_act = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_corrects.double() / len(self.train_loader.dataset)
            self.model.moe.reset_activation_counts()
            self.model.moe.experts[-1].reset_activation_counts()
            val_loss,val_acc = self.evaluate(self.test_loader)
            self.upload_expert_id = torch.argmax(act).item()
            self.logger.info(f"--client: {self.client_id} --epoch:{epoch+1}/{self.client_epochs} --train_loss :{train_loss:.4f} --train_acc :{train_acc:.4f} --val_loss : {val_loss:.4f} --val_acc : {val_acc:.4f}")
            record_dic = {'T': self.c_T, 'client_epoch':epoch+1, 'client_id':self.client_id, "train_loss":train_loss, "train_acc":train_acc.item(), "val_loss":val_loss, "val_acc":val_acc.item()}
            record_result(record_dic=record_dic)

        # other experts --kd-->  global expert
        total_kl = torch.tensor(0.0, device=self.device,requires_grad=True)
        length = len(self.val_loader.dataset)
        for inputs,_ in self.val_loader:
            inputs= inputs.to(self.device)
            _,feature,personal_expert_activations,_ = self.model(inputs)
            expert_prob = personal_expert_activations / personal_expert_activations.sum(dim=-1, keepdim=True)
            g_rep = self.model.moe.experts[-1](feature)
            kl = torch.tensor(0.0, device=self.device)
            for i,expert in enumerate(self.model.moe.experts[:-1]):
                kl = kl + expert_prob[i] * self.kl_loss(expert(feature),g_rep)
            total_kl = total_kl + inputs.shape[0] / length * kl
        torch.nn.utils.clip_grad_norm_(self.model.moe.experts[-1].parameters(), max_norm=1)
        self.optimizer_list[-1].zero_grad()
        total_kl.backward(retain_graph=True)
        self.optimizer_list[-1].step()

        self.save_client_model()
        return self.upload_expert_id,val_acc,shared_act






