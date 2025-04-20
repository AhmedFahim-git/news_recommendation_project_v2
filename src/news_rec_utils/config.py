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
MODEL_PATH = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

NEWS_TEXT_MAXLEN = 100  # Actually close to 600

EMBEDDING_DIM = 1536

REDUCED_DIM = 768

IMPRESSION_MAXLEN = 20  # Actually close to 600

NUM_HIDDEN_LAYERS = 2

NEWS_CLASSIFICATION_PROMPT = "Please analyze the following news article to inform if the user would read the following news article.\nThe news article is: "

TORCH_DTYPE = torch.float32

NUM_WORKERS = 4

if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability(DEVICE)
    if compute_capability >= (8, 0):
        TORCH_DTYPE = torch.bfloat16
    elif compute_capability >= (6, 0):
        TORCH_DTYPE = torch.float16

# CLASSIFICATION_MODEL_BATCH_SIZE = 1024
# ATTENTION_MODEL_BATCH_SIZE = 200

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
