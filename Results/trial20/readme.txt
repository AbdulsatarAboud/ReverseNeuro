using specific channels 7 - 19
Perform normalization per subject. Fix the subject values to be between 0 to 1
removing epoches with > 100mV and less then -100mV
500 training iterations
EEG frequencies 5-15,16,17,18
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=np.exp(-7))