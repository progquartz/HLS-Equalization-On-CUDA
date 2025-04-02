## [Color Equalization On Video Using CUDA](https://github.com/progquartz/HLS-Equalization-On-CUDA/tree/main)

# CUDA & OpenMP 기반 비디오 히스토그램 평활화

이 프로젝트는 비디오 프레임에 대한 히스토그램 평활화(histogram equalization)를 CPU와 GPU에서 각각 최적화된 방법으로 구현한 프로젝트입니다. CUDA를 활용한 GPU 병렬 처리와 OpenMP를 활용한 CPU 멀티스레딩 방식을 비교하여 성능 최적화 및 실시간 영상 처리를 실행했습니다.

## 작업기간
- 2023/05/01 ~ 2023/06/23

## 작업자
- **최정용** : CUDA 기반 평활화 커널 구조화, OpenMP CPU 코드 최적화 진행
- **송육권** : OpenMp Cpu 병렬 처리 부문 조사 및 구조화
	 

## 주요 기능

- **색상 평활화 (Histogram Equalization)**
  - **RGB 평활화:** 각 채널(B, G, R)에 대해 개별적으로 히스토그램 평활화를 수행합니다.
  - **HLS 평활화:** RGB 데이터를 HLS(또는 HSL) 색공간으로 변환 후, 명도(L) 채널에 대해 평활화를 수행하여 색상 왜곡 없이 대비를 개선합니다.
  - **결과 혼합:** HLS 평활화 결과와 RGB 평활화 결과를 가중치(Alpha/Beta)를 사용해 혼합함으로써 보다 최적화된 결과를 도출합니다.

- **CUDA 기반 GPU 가속**
  - CUDA 커널을 통해 각 픽셀 단위의 히스토그램 계산 및 평활화 연산을 병렬로 수행했습니다.
  - **CUDA 커널 구조:**
    - `HLS_histogramEqualization`와 `HLS_Equalization`: HLS 채널에 대한 히스토그램 계산 및 평활화.
    - `RGB_histogramEqualization`와 `RGB_Equalization`: RGB 채널별 히스토그램 계산 및 평활화.
    - `ConvertRGBToHLS`와 `ConvertHLSToRGB`: 색공간 변환을 위한 CUDA 커널.
    - `CustomWeight`: 두 이미지를 지정된 가중치로 혼합하는 연산.

- **OpenMP 기반 CPU 병렬 처리**
  - OpenMP를 활용해 CPU에서 다중 스레드로 히스토그램 계산 및 평활화 작업을 병렬화.
  - Serial과 Parallel 두 방식의 구현으로 성능 차이를 비교할 수 있습니다.
  - 

## 성능
- **성능 측정 결과**
  - 178 Frame(1920 x 1080) 의 영상 기준
  - **Serial의 경우** : 53944.1ms 소모
  - **Parallel 버전의 경우** : 15980.4ms 소모
  - **CUDA CV Converting 버전의 경우** :  24372.3ms 소모.
  - **CUDA RGB 버전의 경우** : 2544.95ms 소모
  - **CUDA HSL/RGB 종합의 경우** : 6127.32ms 소모

## 시스템 요구 사항

- **필수 라이브러리:** OpenCV (영상 처리), CUDA Toolkit (GPU 가속), OpenMP (CPU 병렬 처리)
- **지원 플랫폼:** CUDA가 지원되는 GPU가 장착된 시스템 (Windows/Linux)

## 빌드 및 실행 방법

1. **환경 설정**
   - CUDA Toolkit과 OpenCV가 설치되어 있어야 합니다.
   - CMake 또는 Makefile을 이용해 프로젝트 빌드가 실행됩니다.
