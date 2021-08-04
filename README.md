# Image-classification based on CIFAR-10 dataset
The codes is to complete the assignment 2 of CS231n of staford.
mainly build a Multi-Layer Fully Connected Network  
and a Convolutional Network by numpy and pytorch, respectively
## Implementation of Alexnet
model = nn.Sequential(
    nn.Conv2d(in_channel,channel_1,5,padding=0), # H*W = 28 * 28
    nn.ReLU(inplace=True),
    nn.Dropout2d(p=dropout),
    nn.MaxPool2d(kernel_size=2), # H*W = 14 * 14
    
    nn.BatchNorm2d(channel_1),
    nn.Conv2d(channel_1,channel_2,3,padding=1),# H*W = 14 * 14
    nn.ReLU(inplace=True),
    nn.Dropout2d(p=dropout),
    nn.MaxPool2d(kernel_size=2), # H*W = 7 * 7
    
    nn.BatchNorm2d(channel_2),
    nn.Conv2d(channel_2,channel_3,3,padding=1),  # H*W = 7 * 7
    nn.ReLU(inplace=True),
    nn.Dropout2d(p=dropout),
    nn.Conv2d(channel_3,channel_4,3,padding=1),  # H*W = 7 * 7
    nn.ReLU(inplace=True),
    nn.Dropout2d(p=dropout),
    nn.Conv2d(channel_4,channel_5,3,padding=1),  # H*W = 7 * 7
    nn.ReLU(inplace=True),
    
    Flatten(),
    nn.Linear(channel_5*7*7,hidden_1),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_1,hidden_2),
    nn.ReLU(inplace=True),
    nn.Linear(hidden_2,num_classes) 
)
