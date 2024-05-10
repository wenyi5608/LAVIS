import argparse
import torch
from PIL import Image
from pathlib import Path
import numpy as np

from lavis.models import load_model_and_preprocess
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
import openvino as ov

PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",}

#load image
raw_image = Image.open("../docs/_static/merlion.png").convert("RGB")
caption = "a large fountain spewing water into the air"
device = "cpu"

def sample_blip2_image_text_match_pipeline():
    sample = {"image": image, "text_input": [text_input]}

    features_image = model.extract_features(sample, mode="image")
    features_text = model.extract_features(sample, mode="text")

    print(features_image.image_embeds.shape)
    # torch.Size([1, 32, 768])
    print(features_text.text_embeds.shape)
    # torch.Size([1, 12, 768])

    # low-dimensional projected features
    print(features_image.image_embeds_proj.shape)
    # torch.Size([1, 32, 256])
    print(features_text.text_embeds_proj.shape)
    # torch.Size([1, 12, 256])
    similarity = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:,0,:].t()).max()
    print("torch result: ",similarity)
    # tensor([[0.3642]])

def ov_blip2_image_text_match_pipeline(
        ov_image_feature: ov.CompiledModel,
        ov_text_feature: ov.CompiledModel,
        ov_tokenizer: ov.CompiledModel,):
    request = ov_image_feature.create_infer_request()
    request.start_async(image, share_inputs=False)
    request.wait()
    outputs = {}
    outputs["image_embeds"] = request.get_tensor("image_embeds").data
    outputs["image_embeds_proj"] = request.get_tensor("image_embeds_proj").data
    image_embeds_proj = torch.from_numpy(outputs["image_embeds_proj"])

    ov_text = ov_tokenizer([caption])

    inputs = {}
    inputs["input_ids"] = ov_text["input_ids"]
    inputs["attention_mask"] = ov_text["attention_mask"]
    
    text_request = ov_text_feature.create_infer_request()
    text_request.start_async(inputs, share_inputs=False)
    text_request.wait()
    outputs["text_embeds"] = text_request.get_tensor("text_embeds").data
    outputs["text_embeds_proj"] = text_request.get_tensor("text_embeds_proj").data
    text_embeds_proj = torch.from_numpy(outputs["text_embeds_proj"])

    similarity = (image_embeds_proj @ text_embeds_proj[:,0,:].t()).max()
    print("ov result: ",similarity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Export Blip2 feature extrature Model to IR", add_help=True)
    parser.add_argument("--model_type", "-m", type=str, default="pretrain", required=True, help="path to config file")
    parser.add_argument('-o', '--output_path', default='/tmp/blip2')

    args = parser.parse_args()

    core = ov.Core()
    model_type = args.model_type
    if model_type not in PRETRAINED_MODEL_CONFIG_DICT:
        print(f"Available model types are {[k for k in PRETRAINED_MODEL_CONFIG_DICT.keys()]}")
        exit(0)
    model_path = args.output_path

    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type=model_type, is_eval=True, device=device)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)

    #torch inference
    sample_blip2_image_text_match_pipeline()

    # set path to save openvino IR
    IMAGE_MODEL_OV = Path(f"{model_path}/blip2_{model_type}_image.xml")
    TEXT_MODEL_OV = Path(f"{model_path}/blip2_{model_type}_text.xml")
    TOKENIZER_MODEL_OV = Path(f"{model_path}/blip2_{model_type}_tokenizer.xml")

    # convert image model to openvino IR
    if not IMAGE_MODEL_OV.exists():
        image_model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_extract", model_type=model_type, is_eval=True, device=device)
        ov_image_extract = ov.convert_model(image_model, example_input=torch.randn(1, 3, image.shape[2], image.shape[3]), input=(1, 3, image.shape[2], image.shape[3]))
        ov.save_model(ov_image_extract, str(IMAGE_MODEL_OV), compress_to_fp16=True)

    # convert text model to openvino IR
    if not TEXT_MODEL_OV.exists():
        text_model, vis_processors, text_processors = load_model_and_preprocess("blip2_text_extract", model_type=model_type, is_eval=True, device=device)
        text = text_model.tokenizer(caption, return_tensors="pt", padding=True)
        input_ids = text["input_ids"]
        attention_mask = text["attention_mask"]
        dummy_inputs = (
            input_ids,
            attention_mask,)
        ov_text_extract = ov.convert_model(text_model, example_input=dummy_inputs)
        ov.save_model(ov_text_extract, str(TEXT_MODEL_OV), compress_to_fp16=True)  
    
    # convert tokenizer to openvino IR
    if not TOKENIZER_MODEL_OV.exists():
        truncation_side="right"
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        ov_tokenizer = convert_tokenizer(tokenizer, with_detokenizer=False)
        ov.save_model(ov_tokenizer, str(TOKENIZER_MODEL_OV), compress_to_fp16=True)

    if IMAGE_MODEL_OV.exists():
        ov_image_model = core.compile_model(IMAGE_MODEL_OV, device_name=device.upper())
    
    if TEXT_MODEL_OV.exists():
        ov_text_model = core.compile_model(TEXT_MODEL_OV, device_name=device.upper())

    #currently , tokenizer IR only support CPU compile 
    if TOKENIZER_MODEL_OV.exists():
        ov_tokenizer_model = core.compile_model(TOKENIZER_MODEL_OV, device_name="CPU")

    #openvino inference
    ov_blip2_image_text_match_pipeline(ov_image_feature=ov_image_model, ov_text_feature=ov_text_model, ov_tokenizer=ov_tokenizer_model)


