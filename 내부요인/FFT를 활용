# Python에서 FFT를 이용하여 딥러닝시 GPU의 클럭 특징 감지

## FFT(Fast Fourier Transform)
두 가지 주파수를 이용하여 FFT의 활용을 보여드리겠습니다.

상황은 다음과 같습니다.

1. 1hz와 5hz로 각각 진동하는 주파수가 있습니다.
2. 이 파동이 만나 간섭을 일으킨 파동이 있습니다.
3. 간섭을 일으킨 파동에서 FFT를 진행합니다.

![1](https://github.com/user-attachments/assets/74b28802-52b9-436c-83c3-bdda47a57ddc)

결과로부터 FFT를 활용한다면 신호 내에 어떤 주파수 성분이 있는지 분석할 수 있음을 알 수 있습니다.

이제 실제 딥러닝중인 GPU를 통해 구현해보겠습니다.

딥러닝 중인 GPU의 Load율, Memory Load율, Power Consumption입니다.


![2](https://github.com/user-attachments/assets/e9a52f22-bce6-4fda-89c4-434b3cf1c9b5)

이렇게 보면 어떤 주기가 있는지 전혀 알 수 없습니다.

하지만 각각의 그래프에 대해서 FFT를 적용하고 그래프를 합친다면 주기를 찾아낼 수 있습니다.


![3](https://github.com/user-attachments/assets/7be00673-bb79-42c3-9e04-24fa9587ca65)

2.22HZ가 주기니 약 0.45초의 주기를 가집니다.

그리고 그래프를 봐도 2.5초동안 각각의 그래프가 5번에 가까운 진동을 했습니다.

from https://steady-board-0e7.notion.site/Python-FFT-GPU-fd5f0c3528ca429e8e3c3def71ced112
