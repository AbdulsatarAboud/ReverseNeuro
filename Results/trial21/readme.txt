using specific channels 4 - 19 
removing epoches with > 100mV and less then -100mV
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=np.exp(-7))