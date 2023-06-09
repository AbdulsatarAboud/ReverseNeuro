import numpy as np
import sys
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader

def saturateAmplitudes(data_set):
    data_set[data_set > 10000] = 10000
    data_set[data_set < -10000] = -10000
    return data_set

def balanceClasses(data_set, label_set):
    zero_indices = np.where(label_set == 0)[0]
    one_indices = np.where(label_set == 1)[0]

    max_count = min(len(zero_indices), len(one_indices))

    zero_keep_indices = np.random.choice(zero_indices, size=max_count, replace=False)
    one_keep_indices = np.random.choice(one_indices, size=max_count, replace=False)

    keep_indices = np.sort(np.concatenate((zero_keep_indices, one_keep_indices)))

    return data_set[keep_indices,:,:], label_set[keep_indices]

def includeChannel(data_set, channels):
    return data_set[:,channels,:]

def getDataSamples(EEG_samples, partic_id):

    data = EEG_samples['data'][0,partic_id]
    sick, sick_lables = data['sick'][0][0], []
    non_sick, non_sick_lables = data['non_sick'][0][0], []

    if np.any(sick):
        sick_lables = np.ones((sick.shape[0],1))

    if np.any(non_sick):
        non_sick_lables = np.zeros((non_sick.shape[0], 1))


    if (np.any(sick)) and (np.any(non_sick)):
        epoches = np.concatenate((sick, non_sick), axis=0)
        lables = np.concatenate((sick_lables, non_sick_lables), axis=0)
    elif (np.any(sick)) and (not np.any(non_sick)):
        epoches = sick
        lables = sick_lables
    elif (not np.any(sick)) and (np.any(non_sick)):
        epoches = non_sick
        lables = non_sick_lables

    epoches = saturateAmplitudes(epoches)
    epoches = includeChannel(epoches, range(4,20)) # Uncomment to inclead specifi channels

    return epoches, lables

def normalizeSamples(EEG_samples):

    mean = np.mean(EEG_samples)
    std = np.std(EEG_samples)

    return (EEG_samples - mean)/std

def generateTrainTest(EEG_samples, LOU_subject_id, normalize = False):
    train_data_set, train_label_set = [], []
    no_participants = EEG_samples.shape[1]
    for partic_id in range(0,no_participants):
        if LOU_subject_id != partic_id:
            epoches, lables = getDataSamples(EEG_samples, partic_id)
            if not np.any(train_data_set):
                train_data_set = epoches
                train_label_set = lables
            else:
                train_data_set = np.concatenate((train_data_set, epoches), axis=0)
                train_data_set = np.nan_to_num(train_data_set, nan=0) # replace nan values with 0
                if normalize:
                    train_data_set = normalizeSamples(train_data_set) # normalize the data

                train_label_set = np.concatenate((train_label_set, lables), axis=0)
        else:
            test_data_set, test_label_set = getDataSamples(EEG_samples, partic_id)
            test_data_set = np.nan_to_num(test_data_set, nan=0) # replace nan values with 0
            if normalize:
                test_data_set = normalizeSamples(test_data_set) # normalize the data

            valid_data_set, valid_label_set = getDataSamples(EEG_samples, partic_id)
            valid_data_set = np.nan_to_num(valid_data_set, nan=0) # replace nan values with 0
            if normalize:
                valid_data_set = normalizeSamples(valid_data_set) # normalize the data

    train_data_set, train_label_set = balanceClasses(train_data_set, train_label_set) # for ensuring a 50:50 ratio between sick and non-sick

    data_set = np.concatenate((train_data_set, test_data_set), axis=0)
    label_set = np.concatenate((train_label_set, test_label_set), axis=0)

    return [data_set, label_set], [train_data_set, train_label_set], [test_data_set, test_label_set]


def writeStatsToFile(full_data, train_data, test_data, file_name):

    data_set, label_set = full_data[0], full_data[1]
    train_data_set, train_label_set = train_data[0], train_data[1]
    test_data_set, test_label_set = test_data[0], test_data[1]

    with open(file_name, 'w') as file:
        file.write("-------- INFORMATION ON THE DATA ---------\n")
        file.write("Shapes\n")
        file.write("Train Data: "+str(train_data_set.shape)+" Train Lable: "+str(train_label_set.shape)+"\n")
        file.write("Test Data: "+str(test_data_set.shape)+" Test Lable: "+str(test_label_set.shape)+"\n")

        file.write("***********************************************************************************\n")

        file.write("Number of NANs in the data\n")
        file.write("Train Data: "+str(np.sum(np.isnan(train_data_set)))+" Train Lable: "+str(np.sum(np.isnan(train_label_set)))+"\n")
        file.write("Test Data: "+str(np.sum(np.isnan(test_data_set)))+" Test Lable: "+str(np.sum(np.isnan(test_label_set)))+"\n")

        file.write("***********************************************************************************\n")

        file.write("Some stats on the data\n")
        file.write("Number of 'non sick' samples: "+ str(np.count_nonzero(label_set == 0))+"\n")
        file.write("Count voltage values > 10,000: "+ str(np.count_nonzero(data_set > 10000))+"\n")
        file.write("Count voltage values < -10,000: "+ str(np.count_nonzero(data_set < -10000))+"\n")

        file.write("**************************************** FULL DATASET *******************************************\n")

        file.write("Number of 'sick' samples: "+ str(np.count_nonzero(label_set == 1))+"\n")
        file.write("Number of 'non sick' samples: "+ str(np.count_nonzero(label_set == 0))+"\n")
        file.write("Maximum Value in Data Set: " + str(np.amax(data_set))+"\n")
        file.write("Minimum Value in Data Set: " + str(np.amin(data_set))+"\n")
        file.write("Mean Value in Data Set: " + str(np.mean(data_set))+"\n")
        file.write("Standard Daviation in Data Set: " + str(np.std(data_set))+"\n")

        file.write("************************************* TRAINING DATASET **********************************************\n")

        file.write("Number of 'sick' samples: "+ str(np.count_nonzero(train_label_set == 1))+"\n")
        file.write("Number of 'non sick' samples: "+ str(np.count_nonzero(train_label_set == 0))+"\n")
        file.write("Maximum Value in Training Set: " + str(np.amax(train_data_set))+"\n")
        file.write("Minimum Value in Data Set: " + str(np.amin(train_data_set))+"\n")
        file.write("Mean Value in Data Set: " + str(np.mean(train_data_set))+"\n")
        file.write("Standard Daviation in Data Set: " + str(np.std(train_data_set))+"\n")

        file.write("*************************************** TESTING DATASET ********************************************\n")

        file.write("Number of 'sick' samples: "+ str(np.count_nonzero(test_label_set == 1))+"\n")
        file.write("Number of 'non sick' samples: "+ str(np.count_nonzero(test_label_set == 0))+"\n")
        file.write("Maximum Value in Testing Set: " + str(np.amax(test_data_set))+"\n")
        file.write("Minimum Value in Data Set: " + str(np.amin(test_data_set))+"\n")
        file.write("Mean Value in Data Set: " + str(np.mean(test_data_set))+"\n")
        file.write("Standard Daviation in Data Set: " + str(np.std(test_data_set))+"\n")
        file.close()

    return 0

def appendToFile(info, file):

    with open(file, 'a') as file:
        file.write(info+'\n')
        file.close()

def generateTorchLoaders(data_set, data_label, EEGDataset):
    N = 40 #number of batches
    W = 0 #worker threads
    data_transform = transforms.Compose([transforms.ToTensor()])

    train_set = EEGDataset(data_set=data_set, label_set=data_label, transform=data_transform)
    train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=W)

    return train_loader

# evaluation function
def eval(net, data_loader, file_name=[]):
    loss_function = torch.nn.CrossEntropyLoss()
    # TODO: build your SGD optimizer with learning rate=0.01, momentum=0.9
    # your code here
    optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9, weight_decay=np.exp(-7))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    #net.eval()
    correct = 0.0
    num_images = 0.0
    for i_batch, (images, labels) in enumerate(data_loader):
        labels = torch.reshape(labels, (-1,))
        labels = labels.type(torch.LongTensor)
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()
        outs = net(images.float()) 
        _, preds = outs.max(1)
        correct += preds.eq(labels).sum()
        num_images += len(labels)

    acc = correct / num_images

    if file_name != []:
        appendToFile("Calculated Testing Accuracy: "+str(acc), file_name)

    return acc

# training function
def train(net, train_loader, valid_loader, epoches, file_name):
    loss_function = torch.nn.CrossEntropyLoss()
    # TODO: build your SGD optimizer with learning rate=0.01, momentum=0.9
    # your code here
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=np.exp(-7))
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.002, momentum=0.9, weight_decay=np.exp(-7))
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        print("Cuda is Avaliable")
        
    
    for epoch in range(epoches):
        net.train() 
        correct = 0.0 # used to accumulate number of correctly recognized images
        num_images = 0.0 # used to accumulate number of images
        for i_batch, (images, labels) in enumerate(train_loader):
            t_size = images.size()
            labels = torch.reshape(labels, (-1,))
            labels = labels.type(torch.LongTensor)

            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()
            test_image = images.float()
            test_weights = net.conv1.weight.data
            outputs = net(images.float())
            loss = loss_function(outputs, labels)
            loss.backward()

            optimizer.step()
            
            dummy,predicted = outputs.max(1)
            correct += predicted.eq(labels).sum()
            num_images += len(labels)

        acc = correct / num_images
        acc_eval = eval(net, valid_loader, [])
        print('epoch: %d, lr: %f, accuracy: %f, loss: %f, valid accuracy: %f' % (epoch, optimizer.param_groups[0]['lr'], acc, loss.item(), acc_eval))
        info = "epoch: "+str(epoch)+", lr: "+str(optimizer.param_groups[0]['lr'])+", accuracy: "+str(acc)+", loss: "+str(loss.item())+", valid accuracy: "+str(acc_eval)
        appendToFile(info, file_name)

    return net