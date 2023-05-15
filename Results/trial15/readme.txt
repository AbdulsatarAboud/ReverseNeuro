using specific channels 4 - 19
Perform normalization per subject. Fix the subject values to be between 0 to 1
removing epoches with > 50mV and less then -50mV
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=np.exp(-7))