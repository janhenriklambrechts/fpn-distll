import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import resnet_fpn
import torch.nn.functional as F
from autoaugment import CIFAR10Policy
from cutout import Cutout
from fpn import FPN
from cifar import CIFAR100MultiTransform 
from double_dataset import MultiTransformDataset

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

transform_train_aug = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                         transforms.RandomHorizontalFlip(), CIFAR10Policy(),
                         transforms.ToTensor(), Cutout(n_holes=1, length=16),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(), 
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
multi_transform_trainset = CIFAR100MultiTransform(
         root='/home/lthpc/datasets/data',
        train=True,
        download=False,

        transforms=[transform_train_aug, transform_train])
trainloader = torch.utils.data.DataLoader(
    multi_transform_trainset,
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
    net_teacher = resnet_fpn.resnet50(num_classes=args.class_num, fpn_in_channels=[256, 512, 1024,2048], fpn_out_channels=256, fpn_num_outs=5)
    net_student = resnet_fpn.resnet50(num_classes=args.class_num, fpn_in_channels=[256, 512, 1024,2048], fpn_out_channels=256, fpn_num_outs=5)
 
    print("using resnet 50")



net_teacher.to(device)
#fpn_teacher.to(device)
#fpn_head_teacher.to(device)
net_student.to(device)
#fpn_student.to(device)
#fpn_head_student.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_teacher = optim.SGD(net_teacher.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)
optimizer_student = optim.SGD(net_student.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)


if args.multigpu:
    net = torch.nn.DataParallel(net.cuda())

if __name__ == "__main__":
    best_acc_t = 0
    best_acc_s = 0
    print("Start Training")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(args.epoch):
                teacher_corrects = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                predicteds_teacher = [0, 0, 0, 0, 0, 0, 0]
                student_corrects = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                predicteds_student = [0, 0, 0, 0, 0, 0, 0]
                
                correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
                predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
                if epoch in [80, 160, 240]:
                    for param_group in optimizer_student.param_groups:
                        param_group['lr'] /= 10
                if epoch in [80, 160, 240]:
                    for param_group in optimizer_teacher.param_groups:
                        param_group['lr'] /= 10
               
                net_student.train()
                net_teacher.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data
                    inputs_teacher = inputs[0]
                    inputs_student = inputs[1]
                    if args.multigpu:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    else:
                        inputs_teacher, inputs_student,  labels = inputs_teacher.to(device), inputs_student.to(device), labels.to(device)
                    teacher_outputs = net_teacher(inputs_teacher)
                    student_outputs = net_student(inputs_student)
                                        #   compute loss
                    loss = torch.FloatTensor([0.]).to(device)

                    #   for deepest classifier
                    #loss += criterion(teacher_outputs[0], labels)
                    


                    
                    for index in range(0, len(teacher_outputs)):
                        #teacher_output = outputs[index-1].detach()
                        #teacher_output.requires_grad = False
                        #loss += CrossEntropy(outputs[index], teacher_output) * args.lambda_KD * 9
                        loss += criterion(teacher_outputs[index], labels) # * (1 - args.lambda_KD)
                        teacher_output = teacher_outputs[index].detach()
                        teacher_output.requires_grad = False
                        loss += criterion(student_outputs[index], labels) * (1 - args.lambda_KD) 
                        loss += CrossEntropy(student_outputs[index], teacher_output) * args.lambda_KD * 9

                    #   for faeture align loss


                    optimizer_teacher.zero_grad()
                    optimizer_student.zero_grad()
                    loss.backward()
                    clip_grads(net_student.parameters())
                    clip_grads(net_teacher.parameters())
                    optimizer_teacher.step()
                    optimizer_student.step()

                    total += float(labels.size(0))
                    sum_loss += loss.item()
                    #outputs.append(ensemble)
                    for classifier_index in range(args.classifier+1):
                        #print(outputs[classifier_index].shape)
                        _, predicteds_student[classifier_index] = torch.max(student_outputs[classifier_index].data, 1)

                        student_corrects[classifier_index] += float(predicteds_student[classifier_index].eq(labels.data).cpu().sum())
                        _, predicteds_teacher[classifier_index] = torch.max(teacher_outputs[classifier_index].data, 1)

                        teacher_corrects[classifier_index] += float(predicteds_teacher[classifier_index].eq(labels.data).cpu().sum())
 
                    print('[epoch:%d, iter:%d  Teacher] Loss: %.03f | Acc: Main: %.2f%% FPN_1: %.2f%% FPN_2: %.2f%%  FPN_3: %.2f%% FPN_4: %.2f FPN_5: %.2f'
                           % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                                  100 * teacher_corrects[0] / total, 100 * teacher_corrects[1] / total,
                                                  100 * teacher_corrects[2] / total, 100 * teacher_corrects[3] / total,
                                                  100 * teacher_corrects[4] / total, 100 * teacher_corrects[5] / total))
                    print('[epoch:%d, iter:%d  Student] Loss: %.03f | Acc: Main: %.2f%% FPN_1: %.2f%% FPN_2: %.2f%%  FPN_3: %.2f%% FPN_4: %.2f FPN_5: %.2f'
                           % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                                  100 * student_corrects[0] / total, 100 * student_corrects[1] / total,
                                                  100 * student_corrects[2] / total, 100 * student_corrects[3] / total,
                                                  100 * student_corrects[4] / total, 100 * student_corrects[5] / total))


                print("Waiting Test!")
                with torch.no_grad():
                    corrects_teacher = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    predicteds_teacher = [0, 0, 0, 0, 0, 0, 0]
                    corrects_student = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    predicteds_student = [0, 0, 0, 0, 0, 0, 0]
                

                    total = 0.0
                    for data in testloader:
                        net_teacher.eval()
                        net_student.eval()
                        images, labels = data
                        if args.multigpu:
                            images, labels = images.cuda(), labels.cuda()
                        else:
                            images, labels = images.to(device), labels.to(device)
                        teacher_outputs = net_teacher(images)    
                        student_outputs = net_student(images)
                        #ensemble = sum(outputs) / len(outputs)
                        total += float(labels.size(0))

                        for classifier_index in range(args.classifier+1):
                        #print(outputs[classifier_index].shape)
                            _, predicteds_student[classifier_index] = torch.max(student_outputs[classifier_index].data, 1)

                            corrects_student[classifier_index] += float(predicteds_student[classifier_index].eq(labels.data).cpu().sum())
                            _, predicteds_teacher[classifier_index] = torch.max(teacher_outputs[classifier_index].data, 1)

                            corrects_teacher[classifier_index] += float(predicteds_teacher[classifier_index].eq(labels.data).cpu().sum())
 
                       
                    print('Test Teacher | Acc: Main: %.2f%% FPN_1: %.2f%% FPN_2: %.2f%%  FPN_3: %.2f%% FPN_4: %.2f FPN_5: %.2f'
                           % (
                                                  100 * corrects_teacher[0] / total, 100 * corrects_teacher[1] / total,
                                                  100 * corrects_teacher[2] / total, 100 * corrects_teacher[3] / total,
                                                  100 * corrects_teacher[4] / total, 100 * corrects_teacher[5] / total))
                    print('Test Student | Acc: Main: %.2f%% FPN_1: %.2f%% FPN_2: %.2f%%  FPN_3: %.2f%% FPN_4: %.2f FPN_5: %.2f'
                           % (
                                                  100 * corrects_student[0] / total, 100 * corrects_student[1] / total,
                                                  100 * corrects_student[2] / total, 100 * corrects_student[3] / total,
                                                  100 * corrects_student[4] / total, 100 * corrects_student[5] / total))



                    if corrects_teacher[0]/total > best_acc_t:
                         best_acc_t = corrects_teacher[0]/total
                         torch.save(net_teacher.state_dict(), f'best_teacher_net.pth.tar')
                         torch.save(net_student.state_dict(), f'corresponding_student.pth.tar')
                         print("Best Teacher Accuracy Updated: ", best_acc_t * 100)
                    if corrects_student[0]/total > best_acc_s:
                         best_acc_t = corrects_student[0]/total
                         torch.save(net_student.state_dict(), f'best_student_net.pth.tar')
                         torch.save(net_teacher.state_dict(), f'corresponding_teacher.pth.tar')
                         print("Best Student  Accuracy Updated: ", best_acc_t * 100)


            print("Training Finished, TotalEPOCH=%d" % args.epoch)
print(best_acc_t, best_acc_s)

