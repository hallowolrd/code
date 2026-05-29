import argparse

parse = argparse.ArgumentParser(description="system")

###################################### data settings ################################################
parse.add_argument("--data_name",type=str,default="cifar100",choices=['cifar10', 'cifar100',"tinyimagenet"])
parse.add_argument("--data_path",type=str,default="./data")
parse.add_argument("--train_ratio",type=float,default=0.75)
parse.add_argument("--val_ratio",type=float,default=0.1)
parse.add_argument("--test_ratio",type=float,default=0.15)
parse.add_argument("--data_save_path",type=str,default="./save/data")
parse.add_argument("--batch_size",type=int,default=32)
parse.add_argument("--min_datasize",type=int,default=32)
parse.add_argument("--sample_ratio",type=float,default=0.2)
parse.add_argument("--alpha",type=float,default=0.1)


###################################### base settings ################################################
parse.add_argument("--num_clients",type=int,default=4)
parse.add_argument("--server_epochs",type=int,default=50)
parse.add_argument("--client_epochs",type=int,default=1)
parse.add_argument("--device",type=str,default="cpu")
parse.add_argument("--save_result",type=str,default="./save/result")


###################################### model settings ################################################
parse.add_argument("--topK",type=int,default=2)
parse.add_argument("--num_experts",type=int,default=3)
parse.add_argument("--dropout",type=float,default=0.2)
parse.add_argument("--learning_rate",type=float,default=5e-4)
parse.add_argument("--out_dim",type=int,default=100)
parse.add_argument("--hidden_dim",type=int,default=64)
parse.add_argument("--model_save_path",type=str,default="./save/model")


###################################### other settings ################################################
parse.add_argument("--KLtemperature",type=int,default=2)

parse.add_argument("--max_degree",type=int,default=3)

parse.add_argument("--low_energy_thr",type=float,default=0.5)
parse.add_argument("--tail_energy_thr",type=float,default=0.05)
parse.add_argument("--tail_ratio_thr",type=float,default=0.3)
parse.add_argument("--account",type=float,default=0.85)
