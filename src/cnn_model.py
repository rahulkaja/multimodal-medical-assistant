import torch 
import torch.nn as nn
from torchvision import models

class MedicalImageModel(nn.Module):
    def __init__(self,num_classes=2):
        super(MedicalImageModel,self).__init__()

        # We are going to use a pre-trained ResNet 18 as a strong, Lightweight backbone 
        self.base_model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # We are replacing the fully connected layer with our specific medical tast 
        num_ftrs=self.base_model.fc.in_features
        self.base_model.fc=nn.Linear(num_ftrs,num_classes)

    def forward(self,x):
        """We use this for image classification"""
        return self.base_model(x)
    
    def extract_features(self,x):
        """Used to grab image embeddings for Multimodal Fusion (Phase 4)"""
        modules=list(self.base_model.children())[:-1]
        feature_extractor=nn.Sequential(*modules)

        features=feature_extractor(x)
        return features.view(features.size(0),-1)   
    
if __name__=="__main__":
    model=MedicalImageModel(num_classes=2)
    
    #Create a dummy image tensor (1 image, 3 color channels, 224x224 pixles)
    dummy_image=torch.randn(1,3,224,224)
    output=model(dummy_image)
    features=model.extract_features(dummy_image)

    print(f"Classification Output Shape: {output.shape}")
    print(f"Extracted Features Shape: {features.shape}")



