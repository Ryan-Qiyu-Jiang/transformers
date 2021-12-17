import torch

path_to_caffe_converted = "./resnet101_faster_rcnn_final.caffemodel.pt"
caffe_state = torch.load(path_to_caffe_converted)

path_to_pytorch = "./frcnn_pytorch.pth"
pytorch_state = torch.load(path_to_pytorch)

caffe_list = list(caffe_state.items())
pytorch_list = list(pytorch_state.items())

p = c = 0

new_state = {}
used_keys = []

exclude_list = [
    'rpn_cls_score.weight', 'rpn_cls_score.bias', 'bbox_pred.weight', 'bbox_pred.bias'
]

override_map = {
    'rpn_cls_score.weight': 'proposal_generator.rpn_head.objectness_logits.weight',
    'rpn_cls_score.bias': 'proposal_generator.rpn_head.objectness_logits.bias',
    'bbox_pred.weight': 'roi_heads.box_predictor.bbox_pred.weight',
    'bbox_pred.bias': 'roi_heads.box_predictor.bbox_pred.bias'
}

while c < len(caffe_list) and p < len(pytorch_list):
    c_name = caffe_list[c][0]
    c_data = caffe_list[c][1]
    p_name = pytorch_list[p][0]
    p_data = pytorch_list[p][1]
    
    if c_name in exclude_list:
        c += 1
        continue
    
    if c_data.shape == p_data.shape:
        print(f"Matched {c_name} => {p_name}")
        new_state[p_name] = c_data
        used_keys += [c_name]
        p += 1
        c += 1
    else:
        p += 1


unused_caffe_keys = [ key for key in caffe_state if key not in used_keys ]
unused_pytorch_keys = [ key for key in pytorch_state if key not in new_state and 'tracked' not in key ]

print('unused_caffe_keys')
for key in unused_caffe_keys:
    print(f"{key}: {caffe_state[key].shape}")

print('unused_pytorch_keys')
for key in unused_pytorch_keys:
    print(f"{key}: {pytorch_state[key].shape}")


import pdb;pdb.set_trace()