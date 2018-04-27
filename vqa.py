import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import object_classification as obj_clas




def get_layer(x,word_idx):
    x[word_idx] = 1.0
    return x


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens



class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        # encoder layers
        self.fc1 = nn.Linear(84, 40)
        self.fc2 = nn.Linear(40, 60)
        # Bilinear layer
        self.bil = nn.Bilinear(60,60,60)
        # answer classification network
        self.fc3 = nn.Linear(60, 15)      
        self.fc4 = nn.Linear(15, 10)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x=  self.bil(x,que_rep)
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x



net = obj_clas.Net()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

net.load_state_dict(torch.load('object_pretrained_weights'))

with open('questions.txt') as f:
        corpus = f.read().splitlines()
tokenized_corpus = tokenize_corpus(corpus)
for tokens in tokenized_corpus:
    ques_words = [word for word in tokens if word.isalnum()]
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)



w1=torch.load('w1')
test_corpus = tokenize_corpus(corpus)
lr = 0.001
epoch_count = 2
nnet = NNet()
criterion = nn.CrossEntropyLoss()
que_rep = 0



for epoch in range(1,epoch_count):
    running_loss = 0.0
    optimizer = optim.SGD(net.parameters(), lr= lr,weight_decay = lr/epoch,momentum=0.9, nesterov = False)
    for data in trainloader:
        images, labels = data
        module_output = net(Variable(images))
        for tokens in test_corpus:
            ques_words = [word for word in tokens if word.isalnum()]
            x = torch.zeros(vocabulary_size).float()
            
            # retrieving index
            for word in ques_words:
                index=word2idx.get(word)        
                inp = Variable(get_layer(x,index)).float()

            
            #question representation
            que_rep = torch.matmul(w1,inp) 

            optimizer.zero_grad()
            nnet(module_output)

            loss = criterion(module_output, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

print('Finished Training')


           
       
              
            
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
correct = 0
total = 0
count = 1
for data in testloader:

    images, labels = data
    for tokens in test_corpus:
            ques_words = [word for word in tokens if word.isalnum()]
            x = torch.zeros(vocabulary_size).float()
            # retrieving index
            for word in ques_words:
                index=word2idx.get(word)        
                inp = Variable(get_layer(x,index)).float()
            
            #question representation
            que_rep = torch.matmul(w1,inp) 

            module_output = nnet(Variable(images))
            _, predicted = torch.max(module_output.data, 1)

            print ("Which object is this?")
            print(classes[predicted])
            printf("Is this plane?")
            if(predicted==0):
                print("YES")
            else:
                print("NO")
            printf("Is this car?")
            if(predicted==1):
                print("YES")
            else:
                print("NO")

            printf("Is this bird?")
            if(predicted==2):
                print("YES")
            else:
                print("NO")

            printf("Is this cat?")
            if(predicted==3):
                print("YES")
            else:
                print("NO")

            total += labels.size(0)
            correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


