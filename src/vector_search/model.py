import torch
from typing import Tuple, Optional

from transformers import Blip2Processor, Blip2Model, AutoTokenizer
from sentence_transformers import SentenceTransformer

def load_model(model_name: str) -> Tuple[torch.nn.Module, Optional[torch.nn.Module], Optional[torch.nn.Module]]:
    """
    Load the CLIP model and processor.
    https://github.com/mlfoundations/open_clip

    """
    # Load the BLIP-2 model and processor
    # model = SentenceTransformer(text_encoder_name).to(device)
    #model = AutoModel.from_pretrained(text_encoder_name, 
    #                                  trust_remote_code=True).to(device)
    # Load the BLIP-2 model and processor
    #local_model_path = "./models/blip2-opt-2.7b"
    # model_path = "ViT-B/32"  # "Salesforce/blip2-opt-2.7b"
    # Load BLIP-2 model and processor
    #processor = AutoProcessor.from_pretrained(model_path)
    #model = CLIPModel.from_pretrained(model_path)
    
    if model_name == 'Salesforce/blip2-opt-2.7b':
        print(f"Loading model: {model_name}")
        processor = Blip2Processor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Blip2Model.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        return model, processor, tokenizer
    elif model_name == 'clip-ViT-B-32-multilingual-v1' or model_name == 'clip-ViT-B-32' or model_name == 'clip-ViT-L-14' or model_name == 'clip-vit-large-patch14-336':
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model, None, None
