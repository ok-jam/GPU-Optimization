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

- GPU의 처리 성능과 전력 소비의 균형을 조절하는 것이 중요

  전력 소비 증가는 GPU의 처리 성능을 향상시킬 수 있는 반면, 하드웨어의 열 문제 등 문제를 유발할 수 있고
  그렇기에 성능-전력간 균형을 고려한 GPU 최적화와 효율적 전력관리 기술이 필요

- 다중 요인을 고려하는 GPU 최적화 소프트웨어 개발이 필요

  AI 산업에 도입되는 환경 정책에 대응하기 위해 신재생 에너지 발전량, 지역의 전력 예산,
  탄소 배출량 제한 등 환경-경제적 요인을 고려

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



### 중간평가 (4/22)

중간평가 이전까지 GPU-Z를 활용하여 실시간으로 얻어지는 데이터를 csv파일로 넘겨받아 파이썬의 dash를 통해 기본적인 웹 구현성공

앞으로는 실질적인 그래픽카드(Nvida)의 NVML을 이용하여 웹 구현예정



**중간평가 피드백**

Dash를 이용한 웹구현을 우선시 하기보다 레퍼런스에서 그래픽카드에 우선적으로 적용할 수 있는 알고리즘 

그래픽 카드의 성능을 낮추지만, 에너지 소비량을 줄일 수 있는 최적구간을 찾는 것을 우선적으로 하고, 딥러닝에 맞는 프로그램들을 실행시켜 데이터 얻기



### 진행상황 (9/02)

중간평가 이후, 관련 자료를 활용하여 두 가지 방법을 적용해보았음.

**1) 딥러닝의 특성을 이용하여, 주기 찾기**

 딥러닝은 Epoch룰 수행할 때, GPU 사용량과 Power, Memory등 일정한 주기가 발생
 이를 FFT를 사용하여 노이즈를 줄여 주기를 찾음

 

**2) 파워모델과 성능모델을 이용 SweetSpot 찾기**

 기존에 적용하려했던 방법, 간단하게 각 그래픽카드마다 클럭별 파워소모량과 성능 지표를 활용하여 성능을 측정하고 모델화하여 테스트
 
 테스트 결과 클럭과 연산성능은 선형적 관계를 형성

 가장 전성비가 좋은 구간을 SweetSpot으로 활용

 


