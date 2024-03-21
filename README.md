# GPU - Optimization

4학년 4인 팀 프로젝트

딥러닝을 이용한 전력소비, 탄소 배출량 등 환경적 에너지의 외부적 요인을 적용한 그래픽 카드 최적화 시스템

웹앱과 데이터베이스를 통해 실시간으로 그래픽 카드의 상황을 표시

### 문제 정의서

**딥러닝 학습을 위한 다중 요인 기반 GPU 최적화 시스템 개발**

예상 성과 = GPU 최적화 및 전력관리 데모 응용

**필요기술** 

- 병렬 처리 기술 (Multiprocessing, Cuda)
- 하드웨어 프로파일링 기술 (NVIDIA-smi, Cuda-Profiler, pynvml, psutil)
- WebApp 기술(Dash), 데이터 베이스 관리 기술(Firebase)


**개발 배경 및 필요성** 

- GPU의 처리 성능과 전력 소비의 균형을 조절하는 것이 중요함

  전력 소비 증가는 GPU의 처리 성능을 향상시킬 수 있는 반면, 하드웨어의 열 문제 등 문제를 유발할 수 있고
  그렇기에 성능-전력간 균형을 고려한 **GPU 최적화와 효율적 전력관리 기술**이 필요함

- 다중 요인을 고려하는 GPU 최적화 소프트웨어 개발이 필요

  AI 산업에 도입되는 환경 정책에 대응하기 위해 신재생 에너지 발전량, 지역의 전력 예산,
  탄소 배출량 제한 등 **환경-경제적 요인을 고려**해야 함.

**개발 요구 사항** 

- 비용 예측 모델 개발을 위한 GPU 작업 부하 프로파일링 모듈 개발

  딥러닝 학습의 단계별 메모리 할당량, 전력 소비량, 작업 시간을 측정하는 모듈 구현

- 다중 요인을 고려한 비용 모델링 및 최적화 알고리즘 개발

  신재생 에너지 발전량, 지역의 전력 예산, 탄소 배출량 제한 등의 다중 요인을 가중치로 결합한 비용 모델링 진행
  제약 조건을 포함하는 비선형 최적화 알고리즘 구현

- 딥러닝 학습에서의 GPU 최적화 시스템 구현

  딥러닝 학습과 GPU 최적화 시스템을 병렬 실행이 가능하도록 구현
  학습 반복마다 비용을 최소화하는 최적의 클럭 주파수 탐색 및 적용 기능 구현

- 다중 파라미터 조절이 가능한 인터렉티브 대시보드 개발

  시각화 및 비용 상호작용 (비용 가중치 설정)을 위한 대시보드 구현
  GPU 학습시간 및 전력소비의 실시간 차트 출력
  설정한 비용 가중치를 GPU 최적화 시스템에 즉각적으로 반영

**개발 방향성**
1. 웹 앱을 통해 대시보드 구현
2. 대시보드에 GPU의 실시간 상황을 표시 (Nvida)
3. 여러 제한 상황에 따른 가중치 부여
4. 가중치에 따른 상황에서 GPU 최적화 알고리즘 구현
5. 알고리즘을 적용한 GPU의 실시간 상황을 대시보드에 표시 및 비교

**참고문헌**

About cuda (https://www.infoworld.com/article/3299703/what-is-cuda-parallel-programming-for-gpus.html)

재생에너지 발전을 통한 전력 비용 딥 러닝 처리를 위한 실시간 제어 (https://ieeexplore.ieee.org/abstract/document/8798637 )

에너지 절약에 활용되는 GPU DVFS (https://www.sciencedirect.com/science/article/pii/S2352864816300736)

An Interpretable Machine Learning Model Enhanced Integrated CPU-GPU DVFS Governor (https://dl.acm.org/doi/abs/10.1145/3470974)

Is the powersave governor really saving power? (https://infoscience.epfl.ch/record/307797)

Monitoring Nvidia GPUs using API (https://medium.com/devoops-and-universe/monitoring-nvidia-gpus-cd174bf89311)

탄소공간지도 (https://www.carbonmap.kr/)
