import collections
import torch
from args import parse
import argparse
from Models.CNNs import *
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from client import Client
from utils import *
import random

torch.manual_seed(1)
class Server:
    def __init__(self,parse:argparse.ArgumentParser,logger):
        self.args = parse.parse_args()
        self.num_clients = self.args.num_clients
        self.server_epochs = self.args.server_epochs
        self.batch_size = self.args.batch_size
        self.client_epochs = self.args.client_epochs
        self.clientsID_list = [i+1 for i in range(self.num_clients)]
        self.hidden_dim = self.args.hidden_dim
        self.generate_clients_model()

        self.public_train_loader, self.public_val_loader, self.public_test_loader = None, None, None
        # self.get_public_dataloader()
        self.model_path = self.args.model_save_path + f"/server.pth"
        self.init_global_model()
        self.device = self.args.device

        self.num_experts = self.args.num_experts
        self.topK = self.args.topK

        self.max_degree = self.args.max_degree

        self.low_energy_thr = self.args.low_energy_thr
        self.tail_energy_thr = self.args.tail_energy_thr
        self.tail_ratio_thr = self.args.tail_ratio_thr
        self.account = self.args.account


        self.kldiv = KLDivergenceLoss(temperature=self.args.KLtemperature, reduction="batchmean")
        self.logger = logger
        init_result_csv()
    def get_out_dim(self):
        self.data_name = self.args.data_name
        if self.data_name == "cifar10":
            return 10
        elif self.data_name == "cifar100":
            return 100
        elif self.data_name == "tinyimagenet":
            return 200
        else:
            pass


    def init_global_model(self):
        self.model = GlobalCNN(num_classes=self.get_out_dim(), hidden_dim=self.hidden_dim)
        # self.model = CNNWithMoE_1(num_classes=self.get_out_dim(), hidden_dim=self.hidden_dim)

        self.save_server_model()



    def load_server_model(self):
        return torch.load(self.model_path)

    def load_client_model(self,client_id):
        return torch.load(self.args.model_save_path + f"/{client_id}.pth")

    def save_server_model(self):
        torch.save(self.model, self.args.model_save_path + f"/server.pth")


    def generate_clients_model(self):
        # model_list = [CNNWithMoE_1,CNNWithMoE_2,CNNWithMoE_3,CNNWithMoE_4,CNNWithMoE_5]   # hete
        model_list = [CNNWithMoE_1] # homo
        for id in self.clientsID_list:
            client_model = model_list[id % len(model_list)](num_classes=self.get_out_dim(),hidden_dim=self.hidden_dim)
            model_path = self.args.model_save_path + f"/{id}.pth"
            torch.save(client_model, model_path)




    def train(self):
        self.acc_dict = {}
        all_shared_act = torch.zeros(self.num_experts,device=self.device)
        for c_T in range(self.server_epochs):
            self.logger.info(f"============================== T:{c_T+1} start !!! ===============================\n")
            self.activate_expert = {}
            acc_list = []
            for id in self.clientsID_list:
                upload_expert_id,val_acc,nested_act = Client(parse=parse, client_id=id, logger=self.logger,c_T=c_T).train()
                self.activate_expert[id] = upload_expert_id
                acc_list.append(val_acc)

                if id in self.acc_dict.keys():
                    self.acc_dict[id].append(val_acc.item())
                else:
                    self.acc_dict[id] = [val_acc.item()]

            self.logger.info(f"--average_test_acc :{sum(acc_list) / len(acc_list):.4f} --max_test_acc:{max(acc_list):.4f} --min_test_acc:{min(acc_list):.4f}\n")
            self.aggregation()

            if (c_T + 1) % 2 == 0 and (c_T + 1) >= 2:
                score = self.get_score(self.model.experts["SubMoELayer"])

                # extend
                indices = self.find_extend_indices(lst=score)
                if len(indices) > 0:
                    extend_ids = [random.choice(indices)]
                    extend_queue = self.locate(indices=extend_ids)
                    for queue, path in zip(extend_queue, indices):
                        self.extend_nested_MoE(queue[0], queue[1], path)
                # shrink
                indices = []
                self.find_shrink_indices(score,[],indices)
                for path in indices:
                    self.shrink_moe(path=path)

            self.save_server_model()
            torch.cuda.empty_cache()
    def shrink_moe(self,path):
        shared_expert = self.model.experts["SubMoELayer"]
        shrinking_moe = shared_expert[path[0]]
        if len(path) != 1:
            for i in path[1:]:
                shrinking_moe = shrinking_moe.experts[i]
            weight_list = shrinking_moe.expert_activations
            expert_list = [expert for expert in shrinking_moe.experts]
            weight_list = [score / sum(weight_list) for score in weight_list]
            total_parm = collections.OrderedDict()
            for expert, weight in zip(expert_list, weight_list):
                parm = expert.state_dict()
                for key in parm.keys():
                    if key in total_parm.keys():
                        total_parm[key] += weight * parm[key]
                    else:
                        total_parm[key] = weight * parm[key]
            shrunk_expert = Expert3(self.get_out_dim()).load_state_dict(total_parm)
            shrinking_moe = shrunk_expert

    def is_leaf_moe(self,lst):
        for item in lst:
            if isinstance(item, list):
                return False
        return True

    def find_shrink_indices(self,lst, path, result):
        for i, item in enumerate(lst):
            current_path = path + [i]
            if isinstance(item, list):
                if self.is_leaf_moe(item):
                    if all(d.get('tail_energy') > self.tail_energy_thr for d in item) and all(d.get('tail_ratio') > self.tail_ratio_thr for d in item):
                        result.append(current_path)
                else:
                    self.find_shrink_indices(item, current_path, result)

    def locate(self, indices):
        shared_expert = self.model.experts["SubMoELayer"]
        extend_queue = []
        for path in indices:
            expert = shared_expert
            moe = shared_expert
            for index in path:
                moe = expert
                expert = expert.experts[index]
            extend_queue.append((expert, moe))
        return extend_queue

    def find_extend_indices(self,lst, path=(), indices=None):
        if indices is None:
            indices = []
        for i, value in enumerate(lst):
            current_path = path + (i,)
            if isinstance(value, list):
                self.find_extend_indices(value, current_path, indices)

            elif value["low_energy"] > self.low_energy_thr and value["k99p"] > self.account and len(current_path) <= self.max_degree:
                indices.append(current_path)
        return indices

    def get_score(self,model):
        temp = []
        for expert in model.experts:
            if expert.__class__.__name__ == "SubsubMoELayer":
                score_dic = self.get_score(expert)
            else:
                score_dic = self.weight_matrix_status(W = dict(expert.named_parameters())["fc1.weight"])
            temp.append(score_dic)
        return temp

    def weight_matrix_status(self,W):
        with torch.no_grad():
            s = torch.linalg.svdvals(W.float())
            s2 = s ** 2
            total_energy = s2.sum().item()
            k99 = (torch.cumsum(s2, 0) / total_energy >= 0.99).nonzero()[0].item() + 1
            k99p = k99 / len(s)
            k50 = max(1, int(0.5 * len(s)))
            low_energy = s2[:k50].sum().item() / total_energy
            k90 = (torch.cumsum(s2, 0) / total_energy >= 0.90).nonzero()[0].item() + 1
            tail_energy = s2[k90:].sum().item() / total_energy
            tail_ratio = (len(s) - k90) / len(s)

            return {"k99p":k99p,"low_energy":low_energy,"tail_energy":tail_energy,"tail_ratio":tail_ratio}

    def extend_nested_MoE(self,expert,moe,path):
        nested_moe = SubsubMoELayer(num_experts=6,top_k=4,output_dim=self.hidden_dim)
        nested_moe.gate.load_state_dict(moe.gate.state_dict())
        nested_moe.experts[path[-1]].load_state_dict(expert.state_dict())
        moe.experts[path[-1]] = nested_moe
    def aggregation(self):
        # experts aggregation by avg
        # self.expert_parm_dic = collections.OrderedDict()
        # self.expert_count_dic = collections.OrderedDict()
        # for id in self.clientsID_list:
        #     expert = self.load_client_model(id).moe.experts[self.activate_expert[id]]
        #     parm = expert.state_dict()
        #     name = expert.__class__.__name__
        #
        #     if name in self.expert_parm_dic.keys():
        #         for key in self.expert_parm_dic[name]:
        #             self.expert_parm_dic[name][key] += parm[key]
        #         self.expert_count_dic[name] += 1
        #     else:
        #         self.expert_parm_dic[name] = parm
        #         self.expert_count_dic[name] = 1
        # for name in self.expert_parm_dic.keys():
        #     for key in self.expert_parm_dic[name]:
        #         self.expert_parm_dic[name][key] = self.expert_parm_dic[name][key] / self.expert_count_dic[name]
        # # replace server model
        # for name in self.expert_parm_dic.keys():
        #     self.model.experts[name].load_state_dict(self.expert_parm_dic[name])

        # aggregate shared expert
        total_globalexpert = collections.OrderedDict()
        for id in self.clientsID_list:
            globalexpert = self.load_client_model(id).moe.experts[-1].state_dict()
            for key in globalexpert.keys():
                if key in total_globalexpert.keys():
                    total_globalexpert[key] += globalexpert[key]
                else:
                    total_globalexpert[key] = globalexpert[key]
        for key in total_globalexpert.keys():
            total_globalexpert[key] = total_globalexpert[key] / len(self.clientsID_list)

        self.model.experts["SubMoELayer"].load_state_dict(total_globalexpert)


        # aggregate fe
        total_conv = collections.OrderedDict()
        for id in self.clientsID_list:
            conv = self.load_client_model(id).moe.experts[-1].state_dict()
            for key in total_conv.keys():
                if key in total_conv.keys():
                    total_conv[key] += conv[key]
                else:
                    total_conv[key] = conv[key]
        for key in total_conv.keys():
            total_conv[key] = total_conv[key] / len(self.clientsID_list)

        self.model.conv.load_state_dict(total_conv)






