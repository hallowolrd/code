import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from args import parse


args = parse.parse_args()
data,num_client,alpha = args.data_name,args.num_clients,args.alpha

csv_path = args.save_result + "/detail/" + f'data_{data} client_{num_client} alpha_{alpha}.csv'

class KLDivergenceLoss(nn.Module):
    def __init__(self, temperature=1.0, reduction="batchmean"):
        super(KLDivergenceLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, logits1, logits2):
        log_probs1 = F.log_softmax(logits1 / self.temperature, dim=1)
        probs2 = F.softmax(logits2 / self.temperature, dim=1)

        kl_div = F.kl_div(log_probs1, probs2 + 1e-7, reduction=self.reduction)
        kl_div *= (self.temperature ** 2)
        return kl_div


#
# def logits_smooth(logits,temperature):
#     return logits / temperature

#
# def calculate_communication_cost(tensor: torch.Tensor) -> int:
#     num_elements = tensor.numel()
#     element_size = tensor.storage().element_size()
#     communication_cost = num_elements * element_size
#     return communication_cost / 1048576



def init_result_csv():
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['T', 'client_epoch', 'client_id',"train_loss","train_acc","val_loss","val_acc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def record_result(record_dic:dict):
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['T', 'client_epoch', 'client_id', "train_loss", "train_acc", "val_loss", "val_acc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(record_dic)