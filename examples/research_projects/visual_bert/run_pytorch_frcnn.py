from PIL import Image
import io
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import VisualBertForQuestionAnswering, BertTokenizerFast

# URL = "https://s3.geograph.org.uk/geophotos/06/21/24/6212487_1cca7f3f_1024x1024.jpg"
# OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
# ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
# VQA_URL = "https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt"

# load object, attribute, and answer labels
# objids = utils.get_data(OBJ_URL)
# attrids = utils.get_data(ATTR_URL)
# vqa_answers = utils.get_data(VQA_URL)


# load models and model components
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn_cfg.min_detections = 10 
frcnn_cfg.max_detections = 100

frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
# uncomment to store checkpoint state_dict in pth 
# state = frcnn.state_dict()
# torch.save(state, "./frcnn_pytorch.pth")

image_preprocess = Preprocess(frcnn_cfg)

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# visualbert_vqa = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

winogrond_img_path = "./test_data/ex_0_img_0.jpg"
winogrond_feat_path = "./test_data/ex_0_img_0.npz"

base_image_dir = "/private/home/ryanjiang/winoground/winoground/images_jpg"
image_paths = ["ex_0_img_0", "ex_0_img_1"]
output_path = "/private/home/ryanjiang/winoground/winoground/test_butd_feats"

def run_fcnn(img_path):
    # run frcnn
    images, sizes, scales_yx = image_preprocess(img_path)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    return output_dict

for p in image_paths:
    output = run_fcnn(f"{base_image_dir}/{p}.jpg")
    npz_dict = {
        'features': output['roi_features'][0].cpu().numpy(),
        'norm_bb': output['normalized_boxes'][0].cpu().numpy(),
        'soft_labels': output['obj_probs'][0].cpu().numpy(),
    }
    image_info_path = f"{output_path}/{p}.npz"
    np.savez(image_info_path, **npz_dict)

# pytorch_feat = output_dict['roi_features']
# pytorch_bbox = output_dict['normalized_boxes']

# image_info = {
#     "bbox": output_dict['normalized_boxes'][0].cpu().numpy(),
#     "num_boxes": output_dict['normalized_boxes'][0].size(1),
#     "objects": output_dict['obj_ids'][0].cpu().numpy(),
#     "cls_prob": output_dict['obj_probs'][0].cpu().numpy(),
#     "image_width": output_dict['sizes'][0][1].item(),
#     "image_height": output_dict['sizes'][0][0].item(),
# }

# image_feat = output_dict['roi_features'][0]

# npz_dict = {
#     'features': output_dict['roi_features'][0].cpu().numpy(),
#     'norm_bb': output_dict['normalized_boxes'][0].cpu().numpy()
# }

# output_path = "/private/home/ryanjiang/winoground/winoground/test_butd_feats"
# image_info_path = f"{output_path}/ex_0_img_0.npz"
# np.savez(npz_dict, output_path)

# image_info_dict = np.load(winogrond_feat_path)
# correct_feat = np.array(image_info_dict['features'], dtype='float64')
# correct_bbox = image_info_dict['norm_bb']

# import pdb;pdb.set_trace()

