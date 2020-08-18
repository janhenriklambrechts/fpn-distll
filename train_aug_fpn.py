import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import resnet_bl
import torch.nn.functional as F
from autoaugment import CIFAR10Policy
from cutout import Cutout
from fpn import FPN
from fpn_distill_head import FPNHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--depth', default=50, type=int)
parser.add_argument('--class_num', default=100, type=int)
parser.add_argument('--epoch', default=250, type=int)
parser.add_argument('--lambda_KD', default=0.5, type=float)
parser.add_argument('--classifier', default=5, type=int)
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
                         transforms.RandomHorizontalFlip(), CIFAR10Policy(),
                         transforms.ToTensor(), Cutout(n_holes=1, length=16),
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
    net = resnet_bl.resnet50(num_classes=args.class_num)
    fpn = FPN(in_channels=[256,512,1024,2048], out_channels=256, num_outs=5)
    fpn_head = FPNHead()
    print("using resnet 50")


net.to(device)
fpn.to(device)
fpn_head.to(device)
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
                corrects = [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
                    out, feats = net(inputs)
                    fpn_outs = fpn(feats)
                    #print(len(fpn_outs))
                    head_outs, _ = fpn_head(fpn_outs)
                    #print(len(head_outs))
                    outputs = [out] + list(head_outs)
                    #outputs_g = torch.cat((out, head_outs[0], head_outs[1], head_outs[2], head_outs[3], head_outs[4]))
                    #ensemble = torch.sum(outputs_g, dim=1)/len(outputs_g)
                    #ensemble.detach_()
                    #ensemble.requires_grad = False

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
                    #clip_grads(net.parameters())
                    optimizer.step()

                    total += float(labels.size(0))
                    sum_loss += loss.item()
                    #outputs.append(ensemble)
                    for classifier_index in range(args.classifier+1):
                        #print(outputs[classifier_index].shape)
                        _, predicteds[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                        corrects[classifier_index] += float(predicteds[classifier_index].eq(labels.data).cpu().sum())
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: Main: %.2f%% FPN_1: %.2f%% FPN_2: %.2f%%  FPN_3: %.2f%% FPN_4: %.2f FPN_5: %.2f'
                           % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                                  100 * corrects[0] / total, 100 * corrects[1] / total,
                                                  100 * corrects[2] / total, 100 * corrects[3] / total,
                                                  100 * corrects[4] / total, 100 * corrects[5] / total))

                print("Waiting Test!")
                with torch.no_grad():
                    correct6, correct5, correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0,0, 0
                    predicted6, predicted5, predicted4, predicted3, predicted2, predicted1, predicted0 = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(),torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
                    correct = 0.0
                    total = 0.0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        if args.multigpu:
                            images, labels = images.cuda(), labels.cuda()
                        else:
                            images, labels = images.to(device), labels.to(device)
                        #outputs, feature_loss = net(images)
                        out, feats = net(images)
                        fpn_outs = fpn(feats)
                        #print(len(fpn_outs))
                        head_outs, _ = fpn_head(fpn_outs)
                        #print(len(head_outs))
                        outputs = [out] + list(head_outs)
                       
                       
                         

                       
                        #outputs_g = torch.cat((out, head_outs[0], head_outs[1], head_outs[2], head_outs[3], head_outs[4]))
 
                        #ensemble =torch.sum(outputs_g,dim=1)/len(outputs_g)
                        #ensemble.detach_()
                        #ensemble.requires_grad = False


                        #ensemble = sum(outputs) / len(outputs)
                        _0, predicted0 = torch.max(outputs[0].data, 1)
                        _1, predicted1 = torch.max(outputs[1].data, 1)
                        _2, predicted2 = torch.max(outputs[2].data, 1)
                        _3, predicted3 = torch.max(outputs[3].data, 1)
                        _4, predicted4 = torch.max(outputs[4].data, 1)
                        _5, predicted5 = torch.max(outputs[5].data, 1)
                        #_6, predicted6 = torch.max(outputs[5].data, 1)

                        correct0 += float(predicted0.cuda().eq(labels.data.cuda()).cpu().sum())
                        correct1 += float(predicted1.cuda().eq(labels.data.cuda()).cpu().sum())
                        correct2 += float(predicted2.cuda().eq(labels.data.cuda()).cpu().sum())
                        correct3 += float(predicted3.cuda().eq(labels.data.cuda()).cpu().sum())
                        correct4 += float(predicted4.cuda().eq(labels.data.cuda()).cpu().sum())
                        total += float(labels.size(0))
                        correct5 += float(predicted5.cuda().eq(labels.data.cuda()).cpu().sum())
                        #correct6 += float(predicted6.cuda().eq(labels.data.cuda()).cpu().sum())
                        
                    print('Test Set AccuracyAcc: Main: %.4f%% FPN_1: %.4f%% FPN_2: %.4f%%  FPN_3: %.4f%% FPN_4:%.4f FPN_5: %.4f'
                           % (100 * correct0 / total, 100 * correct1 / total,
                                                 100 * correct2 / total, 100 * correct3 / total,
                                                 100 * correct4 / total, 100 * correct5 / total
                                                 ))
                    if correct0/total > best_acc:
                        best_acc = correct0/total
                        torch.save(net.state_dict(), f'best0_{best_acc * 100}')
                        print("Best Accuracy Updated: ", best_acc * 100)

            print("Training Finished, TotalEPOCH=%d" % args.epoch)
print(best_acc)

