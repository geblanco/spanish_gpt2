# ! virtualenv /data/shared/virtualenvs/poesIA
# ! source /data/shared/virtualenvs/poesIA/bin/activate
# ! pip install transformers tokenizers fastai fastcore==1.0.0
# ! wget https://raw.githubusercontent.com/piegu/fastai-projects/master/nlputils_fastai2.py -O /data/shared/virtualenvs/poesIA/lib/python3.8/site-packages/nlputils_fastai2.py
import torch

from fastai.text.all import Config
from fastai.text.all import *
from nlputils_fastai2 import (
    get_wiki,
    split_wiki,
    get_one_clean_file,
    get_one_clean_csv_file,
    get_num_tokens,
)

from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    MarianMTModel,
    MarianTokenizer,
)

gpu = 1
torch.cuda.set_device(gpu)
print(f"Cuda device: {torch.cuda.current_device()}")

# Get config of paths
config = Config()
lang = "es"
EOF_TOKEN = "<|endoftext|>"


class TransformersTokenizer(Transform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encodes(self, x):
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))

    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


class DropOutput(Callback):
    def after_pred(self):
        self.learn.pred = self.pred[0]


def splitter(model):
    "Split a GPT2 `model` in 3 groups for differential learning rates."
    
    # First layers group : decoder blocks from 0 to 3
    modules = []
    for i in range(4):
        modules.append(model.transformer.h[i])
    groups = [nn.Sequential(*modules)]

    # Second layers group : decoder blocks from 4 to 7
    modules = []
    for i in range(4, 8, 1):
        modules.append(model.transformer.h[i])
    groups = L(groups + [nn.Sequential(*modules)])

    # Third layers group : decoder blocks from 8 to 11
    modules = []
    for i in range(8, 12, 1):
        modules.append(model.transformer.h[i])
    groups = L(groups + [nn.Sequential(*modules)])
    
    # Fourth layers group : embeddings matrices wte and wpe + LayerNorm at the model output
    groups = L(groups + [nn.Sequential(model.transformer.wte,model.transformer.wpe,model.transformer.ln_f)])
    
    return groups.map(params)


def setup_wiki(data_path, lang):
    """
    If get_wiki fails, do:
    
    mkdir -p `wiki_path`
    cd `wiki_path`
    wget -c https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2
    bzip2 -dk ptwiki-latest-pages-articles.xml.bz2

    and rerun get_wiki
    """
    name = f"{lang}wiki"
    wiki_path = data_path/name
    wiki_path.mkdir(exist_ok=True, parents=True)
    get_wiki(wiki_path, lang)
    dest = split_wiki(wiki_path, lang)
    dest = wiki_path/"docs"
    # Size of downloaded data in the docs folder
    num_files, num_tokens = get_num_tokens(dest)
    print(f"{num_files} files - {num_tokens} tokens")
    get_one_clean_file(dest, lang)
    get_one_clean_csv_file(dest, lang)
    text_file_path = wiki_path/f"all_texts_{lang}wiki.txt"
    csv_file_path = wiki_path/f"all_texts_{lang}wiki.csv"
    return text_file_path, csv_file_path


def train_tokenizer(data_path, wiki_text_file_path):
    # ToDo := Load if weights exists, else setup
    tokenizer_en = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer_en.pad_token = tokenizer_en.eos_token
    vocab_size = tokenizer_en.vocab_size
    max_length = 1024

    tokenizer_es = ByteLevelBPETokenizer()
    tokenizer_es.train(
        files=[str(wiki_text_file_path)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOF_TOKEN]
    )
    tokenizer_es.enable_truncation(max_length=max_length)

    tokenizer_es_path = data_path/"BLBPE_tokenizer_es"
    tokenizer_es_path.mkdir(exist_ok=True, parents=True)
    tokenizer_es.save_model(str(tokenizer_es_path))

    tokenizer_es = GPT2TokenizerFast.from_pretrained(
        str(tokenizer_es_path), pad_token=EOF_TOKEN
    )
    tokenizer_es.model_max_length = max_length

    # tokenizer_es = ByteLevelBPETokenizer(
    #     vocab_file=str(tokenizer_es_path/"vocab.json"),
    #     merges_file=str(tokenizer_es_path/"merges.txt"),
    # )
    # tokenizer_es.enable_truncation(max_length=1024)

    # ToDo := is this necessary
    # tokenizer_en.pad_token = tokenizer_en.eos_token
    return tokenizer_en, tokenizer_es


def get_fastai_tokenizer(tr_tokenizer):
    tokenizer_fastai = TransformersTokenizer(tr_tokenizer)
    return tokenizer_fastai


def setup_embeddings(data_path, model_en, tokenizer_en, tokenizer_es):
    # ToDo := Load if weights exists, else setup
    old_wgts = model_en.transformer.get_input_embeddings().weight.clone().detach()
    wgts_m = old_wgts.mean(0)

    new_vocab_size = tokenizer_es.tokenizer.vocab_size
    new_wgts = old_wgts.new_zeros(new_vocab_size, old_wgts.size(1))

    old_vocab = tokenizer_en.tokenizer.get_vocab()
    new_vocab = tokenizer_es.tokenizer.get_vocab()

    same_tokens_list = list()
    different_tokens_list = list()
        
    for w, idx_new in new_vocab.items():
        idx_old = old_vocab.get(w, -1)
        if idx_old >= 0:
            new_wgts[idx_new] = old_wgts[idx_old]
            same_tokens_list.append((w,idx_new))
        else:
            new_wgts[idx_new] = wgts_m
            different_tokens_list.append((w,idx_new))

    new_wte = nn.Embedding(new_vocab_size, old_wgts.size(1))
    new_wte.weight.data = new_wgts
    model_en.transformer.set_input_embeddings(new_wte)
    model_en.lm_head.weight = model_en.transformer.wte.weight

    print("Spanish wte matrix setup done!\n"
        f"We kept {len(same_tokens_list)} embeddings vectors from the English one.\n"
        f"We did not kept {len(different_tokens_list)} embeddings vectors from the "
        "English one (instead, we used the old wte mean vector).\n"
    )

    torch.save(new_wgts, data_path/"new_wte_weights.pt")
    torch.save(same_tokens_list, data_path/"same_tokens_list.pt")
    torch.save(different_tokens_list, data_path/"different_tokens_list.pt")


def get_dataloader(wiki_csv_path, tokenizer_es):
    # ToDo := save/load instead of preprocess al the time
    df = pd.read_csv(wiki_csv_path)
    # ToDo := Work only over a small subset? (1000)
    df_sample = df[:1000]

    num = int(0.8*len(df_sample))

    # ToDo := save data
    idxs = np.random.randint(0, len(df_sample), len(df_sample))
    idxs_train = idxs[:num]
    idxs_val = idxs[num:]

    all_texts = np.concatenate([df_sample.iloc[idxs_train].text.values, df_sample.iloc[idxs_val].text.values])

    splits = [list(idxs_train), list(idxs_val)]
    tls = TfmdLists(all_texts, TransformersTokenizer(tokenizer_es), splits=splits, dl_type=LMDataLoader)

    # ToDo get data if already saved
    # idxs_train = torch.load(path_data/'idxs_train.pt')
    # idxs_val = torch.load(path_data/'idxs_val.pt')

    # all_texts = np.concatenate([df.iloc[idxs_train].text.values, df.iloc[idxs_val].text.values])
    # splits = [list(idxs_train), list(idxs_val)]
    # tls = TfmdLists(all_texts, TransformersTokenizer(tokenizer_es), splits=splits, dl_type=LMDataLoader)

    bs, sl = 8,1024
    dls = tls.dataloaders(bs=bs, seq_len=sl)
    return dls


def fine_tune(dataloader, model_en, data_path):
    learn = Learner(
        dataloader,
        model_en,
        loss_func=CrossEntropyLossFlat(),
        splitter=splitter,
        cbs=[DropOutput],
        metrics=[accuracy, Perplexity()]
    ).to_fp16()

    learn.validate()
    learn.freeze()
    learn.lr_find()
    learn.fit_one_cycle(1, 2e-3)
    learn.save(data_path/"GPT2_es_1epoch_lr2e-3")


def main():
    data_path = config["data_path"]
    wiki_text_path, wiki_csv_path = setup_wiki(data_path, lang)
    tokenizer_en, tokenizer_es = train_tokenizer(data_path, wiki_text_path)
    tokenizer_fastai_en = get_fastai_tokenizer(tokenizer_en)
    tokenizer_fastai_es = get_fastai_tokenizer(tokenizer_es)
    model_en = GPT2LMHeadModel.from_pretrained("gpt2")
    setup_embeddings(data_path, model_en, tokenizer_fastai_en, tokenizer_fastai_es)
    dataloader = get_dataloader(wiki_csv_path, tokenizer_es)
    print('Should fine-tune')
    fine_tune(dataloader, model_en, data_path)


if __name__ == "__main__":
    main()
