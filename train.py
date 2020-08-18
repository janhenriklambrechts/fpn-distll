import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import resnet_slim_shared_head
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--class_num', default=100, type=int)
parser.add_argument('--epoch', default=250, type=int)
parser.add_argument('--lambda_KD', default=0.5, type=float)
parser.add_argument('--classifier', default=4, type=int)
parser.add_argument('--multigpu', action='store_true', default=False)
parser.add_argument('--clip_grad', default=35, type=int)
args = parser.parse_args()
print(args)



def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def clip_grads(params):
    params = list(
        filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm=args.clip_grad, norm_type=2)


BATCH_SIZE = 128
LR = 0.1

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                         transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset, testset = None, None
if args.class_num == 100:
    print("dataset: CIFAR100")
    trainset = torchvision.datasets.CIFAR100(
        root='/home/lthpc/datasets/data',
        train=True,
        download=False,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='/home2/lthpc/datasets/data',
        train=False,
        download=False,
        transform=transform_test
    )
if args.class_num == 10:
    print("dataset: CIFAR10")
    trainset = torchvision.datasets.CIFAR10(
        root='/home/lthpc/datasets/data',
        train=True,
        download=False,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='/home/lthpc/datasets/data',
        train=False,
        download=False,
        transform=transform_test
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

net = None


if args.depth == 50:
    net = resnet_slim_shared_head.resnet50(num_classes=args.class_num)
    print("using resnet 50")


net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

if args.multigpu:
    net = torch.nn.DataParallel(net.cuda())

if __name__ == "__main__":
    best_acc = 0
    print("Start Training")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(args.epoch):
                corrects = [0, 0, 0, 0, 0, 0, 0]
                predicteds = [0, 0, 0, 0, 0, 0, 0]
                correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
                predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
                if epoch in [80, 160, 240]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data

                    if args.multigpu:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    else:
                        inputs, labels = inputs.to(device), labels.to(device)
                    outputs, feature_loss = net(inputs)

                    ensemble = sum(outputs[:-1])/len(outputs)
                    ensemble.detach_()
                    ensemble.requires_grad = False

                    #   compute loss
                    loss = torch.FloatTensor([0.]).to(device)

                    #   for deepest classifier
                    loss += criterion(outputs[0], labels)


                    
                    for index in range(1, len(outputs)):
                        #teacher_output = outputs[index-1].detach()
                        #teacher_output.requires_grad = False
                        #loss += CrossEntropy(outputs[index], teacher_output) * args.lambda_KD * 9
                        loss += criterion(outputs[index], labels) # * (1 - args.lambda_KD)

                    #   for faeture align loss


                    optimizer.zero_grad()
                    loss.backward()
                    clip_grads(net.parameters())
                    optimizer.step()

                    total += float(labels.size(0))
                    sum_loss += loss.item()
                    outputs.append(ensemble)
                    for classifier_index in range(args.classifier+1):
                        _, predicteds[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                        corrects[classifier_index] += float(predicteds[classifier_index].eq(labels.data).cpu().sum())
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                          ' Ensemble: %.2f%%' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                                  100 * corrects[0] / total, 100 * corrects[1] / total,
                                                  100 * corrects[2] / total, 100 * corrects[3] / total,
                                                  100 * corrects[4] / total))

                print("Waiting Test!")
                with torch.no_grad():
                    correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
                    predicted4, predicted3, predicted2, predicted1, predicted0 = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(),torch.zeros(1).cuda()
                    correct = 0.0
                    total = 0.0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        if args.multigpu:
                            images, labeles = images.cuda(), labels.cuda()
                        else:
                            images, labels = images.to(device), labels.to(device)
                        outputs, feature_loss = net(images)
                        ensemble = sum(outputs) / len(outputs)
                        _0, predicted0 = torch.max(outputs[0].data, 1)
                        _1, predicted1 = torch.max(outputs[1].data, 1)
                        _2, predicted2 = torch.max(outputs[2].data, 1)
                        _3, predicted3 = torch.max(outputs[3].data, 1)
                        _4, predicted4 = torch.max(ensemble.data, 1)

                        correct0 += float(predicted0.cuda().eq(labels.data.cuda()).cpu().sum())
                        correct1 += float(predicted1.cuda().eq(labels.data.cuda()).cpu().sum())
                        correct2 += float(predicted2.cuda().eq(labels.data.cuda()).cpu().sum())
                        correct3 += float(predicted3.cuda().eq(labels.data.cuda()).cpu().sum())
                        correct4 += float(predicted4.cuda().eq(labels.data.cuda()).cpu().sum())
                        total += float(labels.size(0))

                    print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
                          ' Ensemble: %.4f%%' % (100 * correct0 / total, 100 * correct1 / total,
                                                 100 * correct2 / total, 100 * correct3 / total,
                                                 100 * correct4 / total))
                    if correct0/total > best_acc:
                        best_acc = correct0/total
                        torch.save(net.state_dict(), f'best0_{best_acc * 100}')
                        print("Best Accuracy Updated: ", best_acc * 100)

            print("Training Finished, TotalEPOCH=%d" % args.epoch)
print(best_acc)

