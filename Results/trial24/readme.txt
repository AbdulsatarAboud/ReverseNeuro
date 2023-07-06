using specific channels 4 - 19 
removing epoches with > 500mV and less then -500mV
normalizing the entire datase from 0 to 1
50-50 split between training and testing data ensuring almost the same number of sick and non-sicj epoches in both the training and testing set
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=np.exp(-7))