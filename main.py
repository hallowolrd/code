from server import *
from args import parse
import warnings
warnings.filterwarnings("ignore")
import logging


args = parse.parse_args()
data,num_client,alpha = args.data_name,args.num_clients,args.alpha

logger_name = args.save_result + "/logs/" + f'data_{data} client_{num_client} alpha_{alpha}'

logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

file_handler = logging.FileHandler(f"{logger_name}.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

logger.addHandler(console_handler)
logger.addHandler(file_handler)




Server(parse=parse,logger = logger).train()