# %% Imports
import numpy as np
import csv
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# %% Useful Methods for loading the stimulus dataset
def _read_log(log_path):
    '''
        Read the .csv file associated with the given path and return each row as a 
        dictionnary whose values are the different column content in regard to the row
        and whose keys are the content of each column of the first row.
    '''
    
    with open(log_path, mode='r', newline='') as log_file:
        logs = list(csv.DictReader(log_file))
        
    return logs

def _read_ParkSpace_log(PS_path):
    '''
        Read a .csv file describing a ParkSpace and return the list of coordinates of each 
        point in this space, formated as (N, ID, FD)
    '''

    PS_Points = _read_log(PS_path)

    ParkSpace_Description = []
    for point in PS_Points:
        ParkSpace_Description.append( (int(point['numerosity']),
                                       int(point['item_diameter']),
                                       int(point['field_diameter'])
                                    ) )
    return ParkSpace_Description

# %% Useful Methods for Preprocessing the Different Datasets to ImageNet Standards
def Nars_Dataset_Preprocessing(img_path):
    '''
        Load & Preprocess the Dot-pattern stimuli from Nars to abide by the ImageNet Standard Input.
    '''
    imageNet_Normalisation = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dot_array  = np.load(img_path) # Binary array (0 or 1) of size (224, 224)
    img_array  = np.array([dot_array]*3)
    img_tensor = torch.Tensor(img_array)
    img_tensor = imageNet_Normalisation(img_tensor) # Normalisation applied to tensor whose values in [0,1]
    return img_tensor.unsqueeze(0) # imageNet pretrained Nets takes (nBatch, 3, H, W) as inputs

def Mask_Dataset_Preprocessing(img_path):
    '''
        Load & Preprocess the Segmentation Mask of our Dataset Stimuli to abide by the ImageNet Standard Input.
    '''
    imageNet_Normalisation = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    bin_array  = np.load(img_path) # Binary array (0 or 1) of size (900, 900)
    img_array  = np.array([bin_array]*3)
    img_tensor = torch.Tensor(img_array)
    img_tensor = imageNet_Normalisation(img_tensor) # Normalisation applied to tensor whose values in [0,1]
    return img_tensor.unsqueeze(0) # imageNet pretrained Nets takes (nBatch, 3, H, W) as inputs

# %% Useful Classes Definition to Extract Networks Features Representation when Performing a Forward Pass on Selected Layers
class ImageNet_Stimuli_PreProcessing(nn.Module):

    def __init__(self):
        super(ImageNet_Stimuli_PreProcessing, self).__init__()
        self._imageNet_preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),  # ToTensor takes values from [0, 255] and output torch.tensor with values in [0, 1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def _load_stimuli(self, path):
        '''
            Load and re-format the image using PIL library so that it can be processed with torch.transforms

            parameters :
            - path : [str] is the /neurospin/unicog/xxx where the .png stimuli was saved

            remarks:
            - when loading our .png stimuli with PIL, an RGBA format is used with the RGB dimension
            being redondant and the A dimension full of 255 (uint8) -> this dimension is discarded.
        '''

        pil_image   = Image.open(path) # PIL format
        numpy_image = np.array(pil_image)[:,:,:3] # discard the Alpha informationless dimension
        input_image = Image.fromarray(numpy_image) # re-format back to PIL

        return input_image

    def forward(self, path):
        '''
            Load the stimuli .png image associated to the given path and preprocess it so that it can
            be use as input of a network which was trained on ImageNet
        '''

        input_image  = self._load_stimuli(path)
        input_tensor = self._imageNet_preprocess(input_image)
        input_batch  = input_tensor.unsqueeze(0) # imageNet pretrained Nets takes (nBatch, 3, H, W) as inputs

        return input_batch

class Layers_FeatureExtractor(nn.Module):
    '''
        Register forward hooks to extract the features for the given layers of the model.

        parameters :
        - model  : [nn.Module] is the Network of interest
        - layers : Iterable[str] contains the name of the layers where to extract the features from
            - These names should follow the naming convention of model.named_modules()
            
    '''

    def __init__(self, model, layers):
        super(Layers_FeatureExtractor, self).__init__()
        self.model = model
        self.model.eval()
        self._features = {layer: torch.empty(0) for layer in layers}
        
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        with torch.no_grad():
            _ = self.model(x)
        return self._flat_features()

    def _flat_features(self):
        for key in self._features.keys():
            self._features[key] = self._features[key].flatten()
        return self._features

class HCNN_FeatureExtractor(Layers_FeatureExtractor):
    '''
        Register forward hooks to extract the features for the given layers of HCNN network.

        parameters :
        - model       : [nn.Module] is the Network of interest
        - layers      : Iterable[str] contains the name of the layers where to extract the features from
        - layers_name : Iterable[str] contains the user-friendly name of the layers
            
        remarks :
        - only a convinience class that renames the layers of named_modules() as a user-friendly version
    '''

    def __init__(self, model, layers, layers_name):

        assert len(layers) == len(layers_name), f'Mismatch of length between the number of selected layers : {len(layers)} and their associated user-friendly name :  {len(layers_name)} given.'

        self.Model_RENAMED_LAYERS = {layers[i]:layers_name[i] for i in range(len(layers))}
        Model_CONVLAYER_NAME = {layers_name[i]:layers[i] for i in range(len(layers))}
        model_layers = [Model_CONVLAYER_NAME[layer_name] for layer_name in layers_name[:len(layers)]]
        
        super(HCNN_FeatureExtractor, self).__init__(model, model_layers)

        self._renamed_features = {layer_name: torch.empty(0) for layer_name in layers_name[:len(layers)]}

    def forward(self, x):
        features = Layers_FeatureExtractor.forward(self, x)
        for key in features.keys():
            self._renamed_features[self.Model_RENAMED_LAYERS[key]] = features[key]
        return self._renamed_features

class TorchHub_FeatureExtractor(HCNN_FeatureExtractor):
    '''
        Register forward hooks to extract the features from some HCNNs network of interest from TorchHub.

        parameters :
        - model_name : [str] corresponds to the name of the model as found in torch.hub.load()
        - mode       : [str] contains the mode use to load the model from torch.hub.load()
            
        remarks :
        - only a convinience class that manages the loading and creation of hooks for our HCNNs of interset.
    '''

    def __init__(self, model_name, mode):

        assert model_name in ['alexnet', 'resnet50', 'vgg16'], 'The given Model is not managed by the current version of this Class.'
        assert mode in  ['pretrained', 'untrained'], 'The version of the Model, pre/un-trained has to be specified.'

        if mode == 'pretrained':
            model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        else:
            torch.manual_seed(0); np.random.seed(0) # ensure reproductibility of random weights initialisation !
            model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=False)

        RENAMED_LAYERS  = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5']

        AlexNet_LAYERS  = ['features.0', 'features.3', 'features.6', 'features.8', 'features.10']
        ResNet50_LAYERS = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
        VGG16_LAYERS    = ['features.0', 'features.7', 'features.14', 'features.21', 'features.28']

        LAYERS = {'alexnet':AlexNet_LAYERS, 'resnet50':ResNet50_LAYERS, 'vgg16':VGG16_LAYERS}

        super(TorchHub_FeatureExtractor, self).__init__(model, LAYERS[model_name], RENAMED_LAYERS)

# %% Useful Methods for Numerosity Decoding
def _load_labels(N, ID, FD, modality, target_scale='Log'):
    '''
        Load the labels centered y associated to a specific modality : N, IA, TA, FA and Spar
    '''

    if modality == 'N':
        y = np.log(N)
    elif modality == 'IA':
        y = np.log(ID**2)
    elif modality == 'TA':
        y = np.log(N*(ID**2))
    elif modality == 'FA':
        y = np.log(np.pi*(FD**2)/4)
    elif modality == 'Spar':
        y = np.log(np.pi*(FD**2)/(4*N))
    elif  modality == 'SzA':
        y = 2*np.log(ID**2) + np.log(N) # log(SzA) = log(IA) + log(TA)
    else:
        y = 2*np.log(np.pi*(FD**2)/4) - np.log(N) # log(Sp) = log(FA) + log(Spar)

    if target_scale == 'Log':
        return y
    else:
        return np.exp(y)

def _compute_park_space_point(N, ID, FD):
    Sp  = np.round(np.log10(np.pi**2 * FD**4 / N) * 10) / 10
    SzA = np.round(np.log10(ID**4 * N) * 10) / 10
    return Sp, SzA

if __name__ == "__main__":
    pass