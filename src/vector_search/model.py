import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel


def load_model(model_name: str) -> torch.nn.Module:
    """
    Load the CLIP model and processor.
    https://github.com/mlfoundations/open_clip
    SigLIP

    """

    if (
        model_name == "clip-ViT-B-32-multilingual-v1"
        or model_name == "clip-ViT-B-32"
        or model_name == "clip-ViT-L-14"
    ):
        model = SentenceTransformer(model_name)
    elif model_name == "jinaai/jina-clip-v1":
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    else:
        raise ValueError(f"Model {model_name} not supported")

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model
