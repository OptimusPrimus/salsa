from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer, \
    RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, \
    CLIPTokenizer, CLIPTextModel
from transformers import AutoModel, AutoTokenizer

def get_sentence_embedding_model(model):

    MODELS = {
        'openai/clip-vit-base-patch32': (CLIPTextModel, CLIPTokenizer, 512),
        'prajjwal1/bert-tiny': (BertModel, BertTokenizer, 128),
        'prajjwal1/bert-mini': (BertModel, BertTokenizer, 256),
        'prajjwal1/bert-small': (BertModel, BertTokenizer, 512),
        'prajjwal1/bert-medium': (BertModel, BertTokenizer, 512),
        'gpt2': (GPT2Model, GPT2Tokenizer, 768),
        'distilgpt2': (GPT2Model, GPT2Tokenizer, 768),
        'bert-base-uncased': (BertModel, BertTokenizer, 768),
        'bert-large-uncased': (BertModel, BertTokenizer, 1024),
        'roberta-base': (RobertaModel, RobertaTokenizer, 768),
        'roberta-large': (RobertaModel, RobertaTokenizer, 1024),
        'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768),
        "distilroberta-base": (RobertaModel, RobertaTokenizer, 768),
        "sentence-transformers/clip-ViT-B-32-multilingual-v1": (AutoModel, AutoTokenizer, 768)
    }

    if 'clip' not in model:
        sentence_embedding_model = MODELS[model][0].from_pretrained(model,
                                                                 add_pooling_layer=False,
                                                                 hidden_dropout_prob=0.2,
                                                                 attention_probs_dropout_prob=0.2,
                                                                 output_hidden_states=False)
    else:
        sentence_embedding_model = MODELS[model][0].from_pretrained(model)

    tokenizer = MODELS[model][1].from_pretrained(model)

    return sentence_embedding_model, tokenizer, MODELS[model][2]