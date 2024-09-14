
## 소비전력
GPU에서 소비전력을 구하기 위해서는 단위 

1. 0.1초 단위로 nvidia-smi를 통해 소비전력을 불러옵니다.
2. 벤치 마크로는  GEMM(General Matrix-Matrix Multiplication)을 사용했습니다.
3. GEMM이 종료될 때 까지의 전체 소비전력을 계산합니다.

아래는 테스트 결과입니다.


![Image](https://github.com/Haenote/GPU-Optimization/assets/4592459/91399e79-6781-4728-a3b8-0b45458cedf7)




![Image](https://github.com/Haenote/GPU-Optimization/assets/4592459/aeb4a87e-da4f-4606-99e3-643bf8eb3b19)

결과를 통해 알 수 있는 사실은 다음과 같습니다. 

- GPU의 클럭과 연산성능은 선형적 관계입니다.
- 테스트 한 GPU(RTX 3070)을 기준으로 전성비의 최대 구간은 1305mhz~1710mhz부근입니다.

from https://steady-board-0e7.notion.site/Python-GPU-Clock-2a9c2578766641c18c0dec292346c04b
