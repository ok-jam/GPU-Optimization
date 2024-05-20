import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import nvidia_smi
import time
import math
import multiprocessing

def print_gpu_info(over_event, interval=1):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # GPU 인덱스 0에 대한 핸들 가져오기
    sumPower = 0
    start = time.time()
    while not over_event.is_set():
        try:
            gpu_info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            clock_info = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_GRAPHICS)
            mem_info = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_MEM)
            power_info = nvidia_smi.nvmlDeviceGetPowerUsage(handle)  # 전력 소비량 가져오기 (단위: 밀리와트)
            sumPower += power_info / 1000.0
            print("GPU 사용량 - GPU: {}%, 메모리: {}%, 그래픽 클럭: {} MHz, 메모리 클럭: {} MHz, 전력 소비량: {}W".format(gpu_info.gpu, gpu_info.memory, clock_info, mem_info, power_info / 1000.0))
        except Exception as e:
            print(f"Error occurred: {e}")

        time.sleep(interval)

    end = time.time()
    result = end - start
    avgPower = sumPower/(result);
    print(f"경과 시간 : {result} 초")
    print(f"총 전력소비량(초당) : {sumPower}W")
    print(f"평균 전력소비량(초당) : {avgPower:.5f}W")

def train_model():
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 데이터셋 다운로드
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # 모델 정의 (예시로 ResNet18 사용)
    resnet18 = models.resnet18()

    # 모델을 GPU로 이동
    resnet18 = resnet18.to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

    # 학습
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # 1000 미니배치마다 손실 출력
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    print('Finished Training')

    # 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = resnet18(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

if __name__ == "__main__":
    # GPU 성능 측정을 위한 이벤트 생성
    over_event = multiprocessing.Event()

    # GPU 성능 측정을 위한 프로세스 생성 및 시작
    gpu_process = multiprocessing.Process(target=print_gpu_info, args=(over_event,))
    gpu_process.start()

    # 학습 실행
    train_model()

    # 딥러닝이 끝난 후 GPU 성능 측정 스레드 종료
    over_event.set()
    gpu_process.join()
