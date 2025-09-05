import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_URLS = {
    "yolo11n.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0/yolo11n.pt"
}

def download_model_if_missing(model_path):
    if not os.path.exists(model_path):
        import urllib.request
        print(f"모델 '{model_path}'이(가) 없습니다. 다운로드 중...")
        url = MODEL_URLS.get(model_path)
        if not url:
            raise FileNotFoundError(f"{model_path} 다운로드 URL을 찾을 수 없습니다.")
        urllib.request.urlretrieve(url, model_path)
        print("모델 다운로드 완료!")

class YOLO11VideoProcessor:
    def __init__(self, model_path='yolo11n.pt', confidence_threshold=0.5, backend='pytorch', half=False, int8=False):
        download_model_if_missing(model_path)
        self.confidence_threshold = confidence_threshold
        self.backend = backend.lower()
        self.model_path = model_path
        self.half = half
        self.int8 = int8
        self.model = self._load_model()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_update_interval = 30

    def _load_model(self):
        print(f"YOLO11n 모델 로딩 중... (backend={self.backend})")
        model = YOLO(self.model_path)

        if self.backend == 'coreml':
            mlpackage_path = model.export(format='coreml', half=self.half, int8=self.int8, nms=False)
            model = YOLO(mlpackage_path, task='detect')
            print("CoreML 모델 로딩 완료!")
        elif self.backend == 'openvino':
            onnx_path = model.export(format='openvino', verbose=False)
            model = YOLO(onnx_path, task='detect')
            print("OpenVINO 모델 로딩 완료!")
        else:
            print("PyTorch 모델 사용")

        print(f"모델 로딩 완료! 신뢰도 임계값: {self.confidence_threshold}")
        return model

    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count % self.fps_update_interval == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.fps_update_interval / elapsed
            self.start_time = time.time()

    def draw_fps(self, frame):
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Model: {self.backend.capitalize()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame

    def draw_detections(self, frame, results):
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    if conf >= self.confidence_threshold:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {conf:.2f}"
                        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - ts[1] - 10), (x1 + ts[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame

    def benchmark_inference(self, test_frames=100, image_size=(640, 640)):
        print(f"\n추론 속도 벤치마크 시작 ({test_frames} 프레임)...")
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        print("워밍업 중...")
        for _ in range(10):
            self.model(dummy_image, verbose=False)

        start = time.time()
        for i in range(test_frames):
            if i % 20 == 0:
                print(f"진행률: {i+1}/{test_frames}")
            self.model(dummy_image, verbose=False)
        end = time.time()

        total_time = end - start
        avg_fps = test_frames / total_time
        avg_infer_ms = total_time / test_frames * 1000
        print(f"\n=== 벤치마크 결과 ({self.backend.capitalize()}) ===")
        print(f"총 처리 시간: {total_time:.2f}s")
        print(f"평균 FPS: {avg_fps:.1f}")
        print(f"평균 추론 시간: {avg_infer_ms:.2f}ms")
        print("=" * 35)

    def process_webcam(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"카메라 {camera_index}를 열 수 없습니다.")
            return
        print("웹캠 처리 시작! 'q'를 누르면 종료됩니다.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model(frame, verbose=False)
                frame = self.draw_detections(frame, results)
                self.calculate_fps()
                frame = self.draw_fps(frame)
                cv2.imshow('YOLO11n Real-time Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_video_file(self, video_path, output_path=None):
        if not os.path.exists(video_path):
            print(f"비디오 파일 '{video_path}'이(가) 없습니다.")
            return

        cap = cv2.VideoCapture(video_path)
        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"비디오 정보: {width}x{height}, {fps_orig:.1f} FPS, {total_frames} 프레임")

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))

        try:
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
                print(f"\r처리 중... {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)", end='')
                results = self.model(frame, verbose=False)
                frame = self.draw_detections(frame, results)
                self.calculate_fps()
                frame = self.draw_fps(frame)
                if out:
                    out.write(frame)
                cv2.imshow('YOLO11n Video Processing', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print("\n처리 완료!")

def compare_models():
    print("=" * 50)
    print("YOLO11n 모델 성능 비교: PyTorch / OpenVINO / CoreML")
    print("=" * 50)
    for backend in ['pytorch', 'openvino', 'coreml']:
        print(f"\n=== {backend.upper()} 벤치마크 ===")
        processor = YOLO11VideoProcessor(model_path='yolo11n.pt', backend=backend)
        processor.benchmark_inference()

def main():
    print("\n선택하세요:")
    print("1. 웹캠 처리 (PyTorch)")
    print("2. 웹캠 처리 (OpenVINO)")
    print("3. 웹캠 처리 (CoreML)")
    print("4. 비디오 파일 처리 (PyTorch)")
    print("5. 비디오 파일 처리 (OpenVINO)")
    print("6. 비디오 파일 처리 (CoreML)")
    print("7. 성능 비교 벤치마크")

    choice = input("선택 (1-7): ").strip()
    backend_map = {'1':'pytorch','2':'openvino','3':'coreml','4':'pytorch','5':'openvino','6':'coreml'}

    if choice in ['1','2','3']:
        processor = YOLO11VideoProcessor(model_path='yolo11n.pt', backend=backend_map[choice])
        cam_idx = input("카메라 인덱스 (기본 0): ").strip()
        cam_idx = int(cam_idx) if cam_idx.isdigit() else 0
        processor.process_webcam(cam_idx)

    elif choice in ['4','5','6']:
        processor = YOLO11VideoProcessor(model_path='yolo11n.pt', backend=backend_map[choice])
        video_path = input("비디오 파일 경로: ").strip()
        out_path = input("출력 파일 경로 (선택, 엔터 스킵): ").strip() or None
        processor.process_video_file(video_path, out_path)

    elif choice == '7':
        compare_models()
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"\n오류 발생: {e}")
        print("\n계속하려면 Enter, 종료하려면 'q' 입력")
        if input().strip().lower() == 'q':
            break
