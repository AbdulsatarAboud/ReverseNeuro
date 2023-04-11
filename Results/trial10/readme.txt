using specific channels 4 - 19
Perform normalization using z-score: Z = (X - mu)/sig. Z0 normalization.
normalize the train set using 'mu' and 'sig' and use the same train normalization parameters on test set
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=np.exp(-7))