import open_clip
import torch
from PIL import Image
import logging


model_name = "ViT-SO400M-14-SigLIP-384"
device = "mps"

try:
    pretrained_models = dict(open_clip.list_pretrained())
    if model_name not in pretrained_models:
        raise ValueError(f"Model {model_name} is not available in pretrained models.")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, device=device, pretrained=pretrained_models[model_name], precision="fp16")
    model.eval()
    model.to(device).float()  # Move model to device and convert to half precision
    logging.debug(f"Model {model_name} loaded and moved to {device}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

try:
    tokenizer = open_clip.get_tokenizer(model_name)
    logging.debug(f"Tokenizer for model {model_name} obtained")
except Exception as e:
    logging.error(f"Error obtaining tokenizer for model {model_name}: {e}")
    raise

# model, _, transform = open_clip.create_model_and_transforms(
#   model_name="coca_ViT-L-14",
#   pretrained="mscoco_finetuned_laion2B-s13B-b90k"
# )

# im = Image.open("/Users/harper/Public/memes/00016840-PHOTO-2023-06-02-08-23-19.jpg").convert("RGB")
# im = transform(im).unsqueeze(0)

with torch.no_grad():
  # generated = model.generate(im)
  # generated = model.generate_text(im)
  #
  text = "hello world"
  text_tokenized = tokenizer(["hello world"]).to(device)
  print(text_tokenized)
  # print(dir(model))
  text_features = model.encode_text(text_tokenized)
  text_features /= text_features.norm(dim=-1, keepdim=True)
  embeddings = text_features.cpu().numpy().tolist()
  logging.debug("Embeddings generated successfully.")
  print(embeddings)


# print(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
