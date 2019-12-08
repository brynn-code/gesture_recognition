from config import HOME

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gesture import GestureDetection

import cnn

# 定义一些超参数
batch_size = 8
learning_rate = 0.001
num_epoches = 20

data_tf = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 数据集的下载器
train_dataset = GestureDetection(root=HOME + "/train/", transform=data_tf)
test_dataset = GestureDetection(root=HOME + "/test/", transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 选择模型
model = cnn.CNN()
# model = net.Activation_Net(28 * 28, 300, 100, 10)
# model = net.Batch_Net(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

out = None
# 训练模型
for epoch in range(21):
    for data in train_loader:
        img, label = data
        # img = img.view(img.size(0), -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("loss: {:.4}".format(loss.data.item()))
    print("epoch: {}, loss: {:.4}".format(epoch, loss.data.item()))
    if epoch > 0 and epoch % 5 == 0:
        torch.save(model, "%sgesture-cnn_skin_gray_%s.pth" % (HOME, epoch))

model = torch.load(HOME + "gesture-cnn_skin_gray_20.pth")
# 模型评估
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()

    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data.item() * label.size(0)
    _, pred = torch.max(out, 1)
    # print(pred)
    # print(label)
    # print("\n")
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print(
    "Test Loss: {:.6f}, Acc: {:.6f}".format(
        eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))
    )
)

