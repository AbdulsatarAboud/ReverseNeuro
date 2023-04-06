import torch.nn as nn

class Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__() 

        # define layers 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)
        
        
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=300, kernel_size=(2,3), padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.25)

        self.conv4 = nn.Conv2d(in_channels=300, out_channels=300, kernel_size=(1,7), padding=1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=(1,2), stride=2)
        self.dropout4 = nn.Dropout(p=0.25)

        self.conv5 = nn.Conv2d(in_channels=300, out_channels=100, kernel_size=(1,3), padding=1)

        self.conv6 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(1,3), padding=1)
        
        self.fc7 = nn.Linear(in_features=20300, out_features=6144) # fillin the appropriate number of input channels = 100xL*W of previous layer

        self.fc8 = nn.Linear(in_features=6144, out_features=num_classes)

        self.softmax9 = nn.Softmax()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)     
        x = self.max_pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)   
        x = self.max_pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)   
        x = self.max_pool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)   
        x = self.max_pool4(x)
        x = self.dropout4(x)

        x = self.conv5(x)

        x = self.conv6(x)

        x = x.reshape(x.shape[0], -1)
        
        x = self.fc7(x)
        
        x = self.fc8(x)

        x = self.softmax9(x)  
        return x