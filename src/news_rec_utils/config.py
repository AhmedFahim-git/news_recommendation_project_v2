from enum import Enum
import torch


class NewsDataset(Enum):
    MINDsmall_train = "MINDsmall_train"
    MINDsmall_dev = "MINDsmall_dev"
    MINDlarge_train = "MINDlarge_train"
    MINDlarge_dev = "MINDlarge_dev"
    MINDlarge_test = "MINDlarge_test"


class DataSubset(Enum):
    WITH_HISTORY = "with_history"
    WITHOUT_HISTORY = "without_history"
    ALL = "all"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_PATH = "Alibaba-NLP/gte-base-en-v1.5"
# MODEL_PATH = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
MODEL_PATH = "nvidia/NV-Embed-v2"
# MODEL_PATH = "intfloat/multilingual-e5-large-instruct"
# MODEL_PATH = "Alibaba-NLP/gte-large-en-v1.5"

NEWS_TEXT_MAXLEN = 1000  # Actually close to 600

EMBEDDING_DIM = 4096

REDUCED_DIM = 768

IMPRESSION_MAXLEN = 1000  # Actually close to 600

NUM_HIDDEN_LAYERS = 2

NEWS_CLASSIFICATION_PROMPT = "Please analyze the following news article to inform if the user would read the following news article.\nThe news article is: "

QUERY_INSTRUCTION = "Instruct: Given a news article that the user has read, retrieve news articles that the user would also read \nQuery: "

TORCH_DTYPE = torch.float32

NUM_WORKERS = 4

# if torch.cuda.is_available():
#     compute_capability = torch.cuda.get_device_capability(DEVICE)
#     if compute_capability >= (8, 0):
#         TORCH_DTYPE = torch.bfloat16
#     elif compute_capability >= (6, 0):
#         TORCH_DTYPE = torch.float16

# CLASSIFICATION_MODEL_BATCH_SIZE = 1024
# ATTENTION_MODEL_BATCH_SIZE = 200

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# Models to  try out
# Alibaba-NLP/gte-large-en-v1.5
# NovaSearch/stella_en_400M_v5  434M param 8192 embed dim (but recommended 1024)  Mean MTEB 69.39
# BAAI/bge-large-en-v1.5  335M param   1024 embed dim    Mean MTEB 65.89   ->  also cls token
# BAAI/bge-base-en-v1.5   109 M param  768 embed dim     Mean MTEB 65.14
# BAAI/bge-small-en-v1.5  33.4 M param   384 embed dim    Mean MTEB 64.30
# avsolatorio/GIST-large-Embedding-v0 based off BAAI/bge-large-en-v1.5   Mean MTEB 66.25
# avsolatorio/GIST-Embedding-v0    based off BAAI/bge-base-en-v1.5    Mean MTEB 65.50
# avsolatorio/GIST-small-Embedding-v0   based off BAAI/bge-small-en-v1.5    Mean MTEB 64.76
# mixedbread-ai/mxbai-embed-large-v1  335M param   1025 preferred dim (512 can be used)    Mean MTEB 66.26
# WhereIsAI/UAE-Large-V1   335M param  1024 embed dim   Mean MTEB 66.40   CLS Token i.e. first token [:,0]  -> first to try
# intfloat/multilingual-e5-large-instruct   560M param  1024 embed dim  Mean MTEB 65.53
