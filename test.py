import open_clip
from dci import metric_dci
from HDF5 import HDF5
dataset=HDF5("1_Object.h5",("factor_name1","factor_name2"),"images")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def rep_func(x):
    feats=model.encode_image(x.to(device))
    feats/=feats.norm(dim=-1, keepdim=True)
    return feats
metric_dci(dataset,rep_func,num_train=60,num_test=9,batch_size=16)