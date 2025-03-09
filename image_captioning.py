from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from PIL import Image
import torch
import json
import os
from tqdm import tqdm
from briarmbg import BriaRMBG
import numpy as np
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
import requests
from segment_anything import sam_model_registry, SamPredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h

def gen_captioning():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_editing_types = sorted(os.listdir("data/EditBench/EditData"))
    with torch.no_grad():
        for edit_type in tqdm(all_editing_types):
            annotation_files = sorted(os.listdir(os.path.join("data/EditBench/EditData", edit_type)))
            annotation_files = [file_name for file_name in annotation_files if file_name.endswith(".json")]
            for ann_file in annotation_files:
                with open(os.path.join("data/EditBench/EditData", edit_type, ann_file), "r") as f:
                    all_data = json.load(f)
                new_annotations = {}
                for key, ann in all_data.items():
                    image_file = ann["image"]
                    img_dir = os.path.join("data/EditBench/EditData", edit_type, "input", image_file)
                    image = Image.open(img_dir).convert("RGB")
                    inputs = processor(image, return_tensors="pt").to(device)
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    ann.update({"src_caption": caption})
                    new_annotations[key] = ann
                
                with open(os.path.join("data/EditBench/EditData", edit_type, ann_file.replace(".json", "_caption.json")), "w") as f:
                    json.dump(new_annotations, f)

@torch.no_grad()
def make_target_prompt():
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat",
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

    all_editing_types = sorted(os.listdir("data/EditBench/EditData"))
    with torch.no_grad():
        for edit_type in tqdm(all_editing_types):
            annotation_files = sorted(os.listdir(os.path.join("data/EditBench/EditData", edit_type)))
            annotation_files = [file_name for file_name in annotation_files if file_name.endswith("caption.json")]
            for ann_file in annotation_files:
                with open(os.path.join("data/EditBench/EditData", edit_type, ann_file), "r") as f:
                    all_data = json.load(f)
                new_annotations = {}
                for key, ann in all_data.items():
                    src_caption = ann["src_caption"]
                    expression = ann["ori_exp"]
                    prompt = f"I have an image and the corresponding prompt. The original prompt is {src_caption}. Now I want to {expression}. You need to keep the words in the original prompt as much as possible. You just try to output the prompt only without '' indicator."
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(device)
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=512
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    ann.update({"edit_caption": response})
                    new_annotations[key] = ann
                
                with open(os.path.join("data/EditBench/EditData", edit_type, ann_file), "w") as f:
                    json.dump(new_annotations, f)

def test_components():
    with open("data/EditBench/EditData/BGReplacement/BGReplacement_caption.json", "r") as f:
        data = json.load(f)
    print(data["1"])

@torch.no_grad()
def make_bg_mask():
    device = "cuda"
    rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
    rmbg = rmbg.to(device=device, dtype=torch.float32)
    json_files = [file_name for file_name in os.listdir("data/EditBench/EditData/BGReplacement") if file_name.endswith("_caption.json")]
    os.makedirs("data/EditBench/EditData/BGReplacement/mask", exist_ok=True)
    for file_name in json_files:
        with open(os.path.join("data/EditBench/EditData/BGReplacement", file_name), "r") as f:
            data = json.load(f)
        for key, annotation in tqdm(data.items()):
            img_file = os.path.join("data/EditBench/EditData/BGReplacement/input", annotation["image"])
            img = np.array(Image.open(img_file).convert("RGB"))
            H, W, C = img.shape
            k = (256.0 / float(H * W)) ** 0.5
            feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
            feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
            alpha = rmbg(feed)[0][0]
            alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
            alpha = alpha.movedim(1, -1)[0]
            alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
            bg_mask = (alpha < 0.1).astype(np.float32)
            bg_mask = (255*bg_mask).astype(np.uint8)
            Image.fromarray(bg_mask[...,0]).save(os.path.join("data/EditBench/EditData/BGReplacement/mask", annotation["image"]))


@torch.no_grad()
def find_main_edit_object_llava_coloralteration():
    model_path = "/home/csgrad/haichaup/Code/LLaVA/llava-v1.5-7b"
    model_base=None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, True, "cuda")
    conv_mode = "v1"
    all_json_files = sorted(os.listdir("data/EditBench/EditData/ColorAlteration/"))
    all_json_files = [file_name for file_name in all_json_files if (file_name.endswith(".json") and "caption" not in file_name)]
    for json_file in tqdm(all_json_files):
        new_annotations = {}
        with open(os.path.join("data/EditBench/EditData/ColorAlteration/", json_file), "r") as f:
            data = json.load(f)
        
        for key, img_info in data.items():
            img_file = os.path.join("data/EditBench/EditData/ColorAlteration/input/", img_info["image"])
            image = load_image(img_file)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            prompt = f"This is the prompt: {img_info['ori_exp']}. What is object we need to change the color of that prompt? Answer using one or few words."
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=10,
                    use_cache=True)
            
            outputs = tokenizer.decode(output_ids[0]).strip()
            if ("<s>" in outputs):
                outputs = outputs.replace("<s> ", "")
            if ("</s>" in outputs):
                outputs = outputs.replace("</s>", "")
            img_info.update({"blended_objects": outputs.lower()})
            new_annotations[key] = img_info
        
        with open(os.path.join("data/EditBench/EditData/ColorAlteration/", json_file.replace(".json", "_caption.json")), "w") as f:
            json.dump(new_annotations, f)

@torch.no_grad()
def make_image_captioning_llava_coloralteration():
    model_path = "/home/csgrad/haichaup/Code/LLaVA/llava-v1.5-7b"
    model_base=None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, True, "cuda")
    conv_mode = "v1"
    all_json_files = sorted(os.listdir("data/EditBench/EditData/ColorAlteration/"))
    all_json_files = [file_name for file_name in all_json_files if file_name.endswith("_caption.json")]
    for json_file in tqdm(all_json_files):
        new_annotations = {}
        with open(os.path.join("data/EditBench/EditData/ColorAlteration/", json_file), "r") as f:
            data = json.load(f)
        
        for key, img_info in data.items():
            img_file = os.path.join("data/EditBench/EditData/ColorAlteration/input/", img_info["image"])
            image = load_image(img_file)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            edit_object = img_info["blended_objects"]
            prompt = f"<image>\n Make the short caption of this photo, including the color description of {edit_object}"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=40,
                    use_cache=True)
            
            outputs = tokenizer.decode(output_ids[0]).strip()
            if ("<s>" in outputs):
                outputs = outputs.replace("<s> ", "")
            if ("</s>" in outputs):
                outputs = outputs.replace("</s>", "")
            img_info.update({"src_prompt": outputs.lower()})

            prompt = f"The original prompt is {outputs.lower()}. {img_info['ori_exp']}. What is the new prompt? Keep the words in the original prompt as much as possible."
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=40,
                    use_cache=True)
            tgt_prompt = tokenizer.decode(output_ids[0]).strip()
            if ("<s>" in tgt_prompt):
                tgt_prompt = tgt_prompt.replace("<s> ", "")
            if ("</s>" in tgt_prompt):
                tgt_prompt = tgt_prompt.replace("</s>", "")
            img_info.update({"edit_prompt": tgt_prompt})
            new_annotations[key] = img_info
        
        with open(os.path.join("data/EditBench/EditData/ColorAlteration/", json_file), "w") as f:
            json.dump(new_annotations, f)

@torch.no_grad()
def make_segmentation_map():
    owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
    owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
    owl_model.to("cuda")
    sam_checkpoint = "sam_checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to("cuda")
    predictor = SamPredictor(sam)
    all_json_files = sorted(os.listdir("data/EditBench/EditData/ColorAlteration/"))
    all_json_files = [file_name for file_name in all_json_files if file_name.endswith("_caption.json")]
    device = "cuda"
    os.makedirs("data/EditBench/EditData/ColorAlteration/mask", exist_ok=True)
    for json_file in tqdm(all_json_files):
        new_annotations = {}
        with open(os.path.join("data/EditBench/EditData/ColorAlteration/", json_file), "r") as f:
            data = json.load(f)
        
        for key, img_info in data.items():
            img_file = os.path.join("data/EditBench/EditData/ColorAlteration/input/", img_info["image"])
            img = Image.open(img_file).convert("RGB")
            blended_objects = img_info["blended_objects"]
            inputs = owl_processor(text=blended_objects, images=img, return_tensors="pt")
            inputs["attention_mask"] = inputs["attention_mask"].to(device)
            inputs["input_ids"] = inputs["input_ids"].to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            outputs = owl_model(**inputs)
            target_sizes = torch.Tensor([img.size[::-1]])
            results = owl_processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)
            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
            try:
                idx = torch.argmax(scores)
            except:
                continue
            boxes = boxes[idx, :].unsqueeze(0).cpu().numpy().astype(np.float32)
            boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, img.width)
            boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, img.height)
            predictor.set_image(np.array(img))
            try:
                masks, sam_scores, logits = predictor.predict(
                    box=boxes[0:1],
                    multimask_output=False
                )
                mask = masks[0].astype(np.float32)
                mask = (255*mask).astype(np.uint8)
                Image.fromarray(mask).save(os.path.join("data/EditBench/EditData/ColorAlteration/mask", img_info["image"]))
                img_info.update({"mask": img_info["image"]})

            except:
                img_info.update({"mask": "no mask"})
                continue
            new_annotations[key] = img_info
        
        with open(os.path.join("data/EditBench/EditData/ColorAlteration/", json_file), "w") as f:
            json.dump(new_annotations, f)

def find_edit_word_llava_coloralteration():
    model_path = "/home/csgrad/haichaup/Code/LLaVA/llava-v1.5-7b"
    model_base=None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, True, "cuda")
    conv_mode = "v1"
    all_json_files = sorted(os.listdir("data/EditBench/EditData/ColorAlteration/"))
    all_json_files = [file_name for file_name in all_json_files if file_name.endswith("_caption.json")]
    for json_file in tqdm(all_json_files):
        new_annotations = {}
        with open(os.path.join("data/EditBench/EditData/ColorAlteration/", json_file), "r") as f:
            data = json.load(f)
        
        for key, img_info in data.items():
            img_file = os.path.join("data/EditBench/EditData/ColorAlteration/input/", img_info["image"])
            image = load_image(img_file)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            edit_object = img_info["blended_objects"]
            expression = img_info["ori_exp"]
            prompt = f"We need to {expression}. What is the final color of {edit_object}? Answer shortly the color only."
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids, 
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=40,
                    use_cache=True)
            
            outputs = tokenizer.decode(output_ids[0]).strip()
            if ("<s>" in outputs):
                outputs = outputs.replace("<s> ", "")
            if ("</s>" in outputs):
                outputs = outputs.replace("</s>", "")
            img_info.update({"edit_word": outputs.lower() + f" {edit_object}"})
            new_annotations[key] = img_info

        with open(os.path.join("data/EditBench/EditData/ColorAlteration/", json_file), "w") as f:
            json.dump(new_annotations, f)

# find_main_edit_object_llava_coloralteration()
# make_image_captioning_llava_coloralteration()
# find_edit_word_llava_coloralteration()
@torch.no_grad()
def find_main_edit_object_llava_object_removal():
    model_path = "/home/csgrad/haichaup/Code/LLaVA/llava-v1.5-7b"
    model_base=None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, True, "cuda")
    conv_mode = "v1"
    all_json_files = sorted(os.listdir("data/EditBench/EditData/ObjectRemoval/"))
    all_json_files = [file_name for file_name in all_json_files if (file_name.endswith(".json") and "caption" not in file_name)]
    for json_file in tqdm(all_json_files):
        new_annotations = {}
        with open(os.path.join("data/EditBench/EditData/ObjectRemoval/", json_file), "r") as f:
            data = json.load(f)
        
        for key, img_info in data.items():
            img_file = os.path.join("data/EditBench/EditData/ObjectRemoval/input/", img_info["image"])
            image = load_image(img_file)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            prompt = f"{img_info['ori_exp']}. What is object we need to remove? Answer using one or few words."
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=10,
                    use_cache=True)
            
            outputs = tokenizer.decode(output_ids[0]).strip()
            if ("<s>" in outputs):
                outputs = outputs.replace("<s> ", "")
            if ("</s>" in outputs):
                outputs = outputs.replace("</s>", "")
            img_info.update({"blended_objects": outputs.lower()})
            new_annotations[key] = img_info
        
        with open(os.path.join("data/EditBench/EditData/ObjectRemoval/", json_file.replace(".json", "_caption.json")), "w") as f:
            json.dump(new_annotations, f)

@torch.no_grad()
def make_image_captioning_object_removal_llava():
    model_path = "/home/csgrad/haichaup/Code/LLaVA/llava-v1.5-7b"
    model_base=None
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, True, "cuda")
    conv_mode = "v1"
    all_json_files = sorted(os.listdir("data/EditBench/EditData/ObjectRemoval/"))
    all_json_files = [file_name for file_name in all_json_files if file_name.endswith("_caption.json")]
    for json_file in tqdm(all_json_files):
        new_annotations = {}
        with open(os.path.join("data/EditBench/EditData/ObjectRemoval/", json_file), "r") as f:
            data = json.load(f)
        
        for key, img_info in data.items():
            img_file = os.path.join("data/EditBench/EditData/ObjectRemoval/input/", img_info["image"])
            image = load_image(img_file)
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            edit_object = img_info["blended_objects"]
            prompt = f"<image>\n Make the short caption with this photo, including using {edit_object}."
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=40,
                    use_cache=True)
            
            outputs = tokenizer.decode(output_ids[0]).strip()
            if ("<s>" in outputs):
                outputs = outputs.replace("<s> ", "")
            if ("</s>" in outputs):
                outputs = outputs.replace("</s>", "")
            img_info.update({"src_prompt": outputs.lower()})

            prompt = f"The original prompt is {outputs.lower()}. Delete {edit_object} from the prompt. What is the new prompt?"
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=40,
                    use_cache=True)
            tgt_prompt = tokenizer.decode(output_ids[0]).strip()
            breakpoint()
            if ("<s>" in tgt_prompt):
                tgt_prompt = tgt_prompt.replace("<s> ", "")
            if ("</s>" in tgt_prompt):
                tgt_prompt = tgt_prompt.replace("</s>", "")
            img_info.update({"edit_prompt": tgt_prompt})
            new_annotations[key] = img_info
        
        with open(os.path.join("data/EditBench/EditData/ObjectRemoval/", json_file), "w") as f:
            json.dump(new_annotations, f)


# find_main_edit_object_llava_object_removal()
make_image_captioning_object_removal_llava()