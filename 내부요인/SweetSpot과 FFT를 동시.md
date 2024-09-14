
## GPU DVFS

우선 3가지 경우에 대해서 테스트 했습니다.

- Default : GPU의 기본 Governor
- Sweetspot : 앞에서 찾아본 Sweetspot에 해당하는 클럭으로 고정시킨 Governor
- FFT : GPU 동작 특성을 FFT를 통해 주기를 찾아 주기에 맞춰서 랜덤하게 최대 성능과 Sweetspot에 해당하는 클럭을 번갈아 가며 동작시키는 Governor


![3](https://github.com/user-attachments/assets/a8654124-3940-4e51-a758-452377761987)



![4](https://github.com/user-attachments/assets/b9fbe49e-5b0e-4e2e-8700-190db657e003)

일반적인 Governor에서는 전력소비를 줄일 구간이 없는 최대 부하를 가할 수 있는 상태에서의 실험이었습니다.



![5](https://github.com/user-attachments/assets/40d16f41-2761-47a1-8c87-5cdf04508ba7)

위 실험에서의 결과로 알 수 있는 점은 다음과 같습니다.

- SweetSpot 클럭에 고정시키는 것이 가장 전력 효율적입니다.
- 최대 클럭으로 고정시키는 것이 가장 성능 효율적입니다.
- FFT를 통해 주기에 맞게 변경시키는 것은 위의 두 특징을 가져옵니다.


![6](https://github.com/user-attachments/assets/2eb0601d-d66c-48c9-9156-95c87e07717e)

GEMM 벤치마크를 돌려도 동일한 결과를 얻을 수 있었습니다.

from https://steady-board-0e7.notion.site/FFT-GPU-SweetSpot-GPU-DVFS-9a8868f6a38248f8afe518410e805bf0
