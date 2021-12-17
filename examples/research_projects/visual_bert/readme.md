## Running butd feature extraction from pytorch

### overview
What are we trying to do? We want to be able to reproduce butd feature extraction without the caffe and python 2 dependencies from pytorch.  
So there exists a frcnn model in hugging face, and we want to convert the checkpoint from the caffe model into the pytorch hugging face model.  
The main issue is that the weights don't really align, more detail below this is likely just a config change for the pytorch model.  
The features will not be exactly the same as what is extracted a past model as the forward passes are stochastic, however they should be similar up to reordering, and models should have exact accuracy scores between the two models. This is the goal. Almost all the work is done for this but needs about <1 day of work. 



Official? repo https://github.com/peteanderson80/bottom-up-attention

caffe model from https://storage.googleapis.com/up-down-attention/resnet101_faster_rcnn_final.caffemodel

to load these weights convert them to a pytorch state dict inorder to load them

use https://github.com/vadimkantorov/caffemodel2pytorch to do the conversion

modified conversion script to add 
```
	elif args.output_path.endswith('.pt'):
		blobs['heatmap_scale.weight']['shape'] = [1]
		torch.save({(print(k)) or k : torch.FloatTensor(blob['data']).view(*blob['shape']) for k, blob in blobs.items()}, args.output_path)
```
`python -m caffemodel2pytorch resnet101_faster_rcnn_final.caffemodel`

to make it work for this caffe model, this generates `resnet101_faster_rcnn_final.caffemodel.pt`

`/transformers/examples/research_projects/visual_bert` has an example of a similar frcnn model, however the weights do not exactly match. 

`python run_pytorch_frcnn.py` is an example of running the pytorch frcnn model on images from winoground.  
I picked this cus we have the butd features for these pairs which we used for uniter and villa predictions.  
The generated features will be different as frcnn regions have randomness.  
However they should be similar and have similar forward passes.   

To attempt to match the state-dicts run, `python state_dict.py`. 
You will need to run the `run_pytorch_frcnn` script to generate a state_dict for the hugging face frcnn model.

### unused_caffe_keys
```
rpn_cls_score.weight: torch.Size([24, 512, 1, 1])
rpn_cls_score.bias: torch.Size([24])
bbox_pred.weight: torch.Size([6404, 2048])
bbox_pred.bias: torch.Size([6404])
heatmap_conv1.weight: torch.Size([16, 1, 3, 3])
heatmap_conv1.bias: torch.Size([16])
heatmap_conv2.weight: torch.Size([32, 16, 3, 3])
heatmap_conv2.bias: torch.Size([32])
heatmap_conv3.weight: torch.Size([32, 32, 3, 3])
heatmap_conv3.bias: torch.Size([32])
heatmap_embedding.weight: torch.Size([64, 8192])
heatmap_embedding.bias: torch.Size([64])
heatmap_scale.weight: torch.Size([1])
fc_rel1.weight: torch.Size([512, 2368])
fc_rel1.bias: torch.Size([512])
fc_subject.weight: torch.Size([256, 512])
fc_subject.bias: torch.Size([256])
fc_object.weight: torch.Size([256, 512])
fc_object.bias: torch.Size([256])
rel_merge.weight: torch.Size([256, 256])
rel_merge.bias: torch.Size([256])
rel_score.weight: torch.Size([21, 256])
rel_score.bias: torch.Size([21])
```
### unused_pytorch_keys
```
proposal_generator.anchor_generator.cell_anchors.0: torch.Size([12, 4])
proposal_generator.rpn_head.objectness_logits.weight: torch.Size([12, 512, 1, 1])
proposal_generator.rpn_head.objectness_logits.bias: torch.Size([12])
roi_heads.box_predictor.bbox_pred.weight: torch.Size([6400, 2048])
roi_heads.box_predictor.bbox_pred.bias: torch.Size([6400])
```

The proposal generator sizes are 12 instead of 24, we can change this through config or code haven't looked
rpn_cls_score.weight -> proposal_generator.rpn_head.objectness_logits.weight
bbox_pred.weight -> roi_heads.box_predictor.bbox_pred.weight
bbox_pred.bias -> roi_heads.box_predictor.bbox_pred.bias


I just don't know what proposal_generator.anchor_generator.cell_anchors.0 is
will have to look through the code.  

Then we need to verify that the model outputs (caffe vs pytorch) are equivalant.