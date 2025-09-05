# YOLO11n Video Processor

YOLO11n Video Processor는 Ultralytics YOLO11n 모델을 활용하여 웹캠 또는 비디오 파일의 객체 탐지를 실시간으로 수행하고, PyTorch, OpenVINO, CoreML 등 다양한 백엔드를 지원하며, FPS 표시 및 벤치마크 기능을 제공하는 Python 기반 도구입니다.

---

## 주요 기능

1. **모델 자동 다운로드**
   - 지정한 경로에 모델이 없으면 `MODEL_URLS`에서 YOLO11n 모델을 자동 다운로드합니다.
   - 지원 모델: `yolo11n.pt`

2. **다중 백엔드 지원**
   - **PyTorch**: 기본 백엔드로 CPU/GPU에서 실행
   - **OpenVINO**: Intel CPU/Edge 장치 최적화
   - **CoreML**: macOS/iOS 환경 최적화
   - 백엔드 변경 시 모델을 자동으로 해당 형식으로 변환 후 로딩

3. **실시간 FPS 계산 및 화면 표시**
   - 영상 처리 중 프레임당 FPS를 계산하고 화면 상단에 표시
   - 현재 사용 중인 모델 백엔드 이름 표시
   - FPS 계산은 `fps_update_interval` 단위로 갱신

4. **객체 탐지 및 시각화**
   - YOLO11n 모델을 이용해 객체 탐지 수행
   - 바운딩 박스, 클래스명, 신뢰도 표시
   - 신뢰도 임계값(`confidence_threshold`) 설정 가능
   - 바운딩 박스와 텍스트는 OpenCV로 영상 위에 그려짐

5. **웹캠 처리**
   - `process_webcam()` 함수로 웹캠 실시간 객체 탐지
   - 'q' 키 입력으로 종료 가능
   - 카메라 인덱스 지정 가능 (기본 0)

6. **비디오 파일 처리**
   - `process_video_file()` 함수로 로컬 비디오 파일 처리
   - 탐지 결과를 새 비디오로 저장 가능
   - 처리 진행률 및 FPS 표시
   - 'q' 키 입력으로 중간 종료 가능

7. **추론 벤치마크**
   - `benchmark_inference()` 함수로 모델의 평균 FPS 및 추론 시간 측정
   - 워밍업 후 지정한 프레임 수만큼 추론 실행
   - PyTorch / OpenVINO / CoreML 간 성능 비교 가능

8. **모델 성능 비교**
   - `compare_models()` 함수로 모든 백엔드(Pytorch, OpenVINO, CoreML) 벤치마크 자동 실행

---

## 설치 및 요구사항

- Python 3.8 이상
- 필수 라이브러리
```
pip install opencv-python numpy ultralytics
```

---

## 실행
```
python yolo11_video_processor.py
```
