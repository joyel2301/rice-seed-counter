# 볍씨 카운터 (Rice Seed Counter)

스마트폰(DroidCam)이나 USB 웹캠으로 볍씨를 실시간으로 세고, 식물체 라인별로 개수를 기록하는 Windows 프로그램입니다.

## 기능
- **실시간 계수**: 카메라 화면의 볍씨를 실시간으로 검출합니다. 인텔 내장 그래픽(OpenVINO GPU)으로 약 15 fps.
- **라인 기록**: 라인 이름을 입력하고 **다음**(Enter)을 누르면 현재 개수가 그 라인 합계에 더해집니다. CSV로 자동 저장됩니다.
- **사진 저장**: 다음을 누를 때 박스와 개수가 찍힌 사진을 같이 저장합니다.
- **검출 영역**: 화면에서 드래그한 영역 안의 볍씨만 셉니다.
- **입력 크기 자동**: 낟알이 찍힌 크기를 보고 모델 입력 크기(640/960/1280)를 고릅니다.
- **사진 파일 분석**: 저장된 사진을 불러와 셀 수 있습니다.

## 실행
```powershell
riceseedcounter-env\python.exe scripts\rice_counter.py
```

처음 설치할 때는 다음 명령으로 환경을 만듭니다.
```powershell
conda create -p riceseedcounter-env python=3.11 -y
riceseedcounter-env\python.exe -m pip install -r requirements.txt
```

## 배포용 빌드 (파이썬 없는 PC용)
```powershell
powershell -ExecutionPolicy Bypass -File packaginguild.ps1
```
결과물은 `dist\RiceSeedCounter.zip`입니다. exe 옆에 `사용법.txt`, `LICENSE.txt`, `licenses\`(제3자 고지)가 함께 들어갑니다.
빌드를 확인하려면 카메라 없이 사진 한 장으로 자가진단을 돌립니다.
```powershell
dist\RiceSeedCounter\RiceSeedCounter.exe --selftest 사진.jpg 결과.txt
```

## 폴더 구성
| 경로 | 내용 |
|---|---|
| `scripts/rice_counter.py` | GUI (PySide6) |
| `scripts/core.py` | 엔진 로딩, 검출, 영역, 입력 크기 자동 선택, 카메라 찾기 |
| `scripts/ov_backend.py` | OpenVINO 실행기 (인텔 CPU/GPU, 전처리·NMS 포함) |
| `scripts/camera_names.py` | Windows 카메라 이름 목록 |
| `scripts/line_records.py` | 라인 기록 CSV 저장 |
| `scripts/convert_openvino.py` | (개발용) best.pt → OpenVINO 변환. PyTorch 필요 |
| `scripts/inspect_model.py` | (개발용) best.pt 안의 구조 보기. PyTorch 필요 |
| `scripts/extra_modules_compat.py` | (개발용) best.pt를 PyTorch로 열 때 필요한 특수 레이어 |
| `external_models/rice_grain_phenotyping/` | 원본 모델 `best.pt`와 OpenVINO 변환본 9개 |
| `packaging/사용법.txt` | 배포 zip에 넣는 사용 안내 |
| `packaging/build.ps1` | 빌드 + 안내문·라이선스 복사 + zip |
| `licenses/` | 제3자 라이선스 전문과 고지 (`THIRD_PARTY_NOTICES.txt`) |
| `LICENSE.txt` | 이 프로그램 코드의 라이선스 (MIT) |
| `scripts/app_info.py` | 프로그램 이름, 버전, 만든 사람 |
| `assets/` | 앱 아이콘 |
| `results/` | 기록 CSV와 저장된 사진 (실행하면 생김) |

## 엔진
| 엔진 | 설명 |
|---|---|
| OpenVINO GPU FP16 | 인텔 내장 그래픽, 가장 빠름 (기본값) |
| OpenVINO GPU FP32 | 조금 느리지만 PyTorch와 계산이 같음 |
| OpenVINO CPU | 인텔 GPU가 없을 때. 사진 파일 분석에도 사용 |

프로그램은 PyTorch 없이 OpenVINO만으로 돌아갑니다. 입력 크기(640/960/1280)마다 가로(16:9)·정사각형·세로(9:16)
변환본이 있어서, 화면이나 사진 비율에 맞는 것을 자동으로 고릅니다. 전처리와 겹친 박스 정리(NMS)는
Ultralytics와 같은 규칙으로 구현해서 결과가 같습니다.

## 만든 사람
Kim Gikang (Dept. of Plant Medicine, Jeonbuk National University)

이 프로그램은 Anthropic의 AI 도구 Claude의 도움을 받아 작성되었습니다.

## 모델과 라이선스
- 프로그램 코드: MIT License (`LICENSE.txt`)
- 검출 모델 `best.pt`: [BearHuntingExpert/Rice-Grain-Phenotyping-System](https://github.com/BearHuntingExpert/Rice-Grain-Phenotyping-System) (MIT License). 논문 "An integrated high-throughput rice grain phenotyping method for breeding applications"(출판 예정)의 모델입니다.
- 배포 프로그램에 들어가는 라이브러리: OpenVINO(Apache-2.0), OpenCV(Apache-2.0), NumPy(BSD), PySide6/Qt(LGPL-3.0), Python(PSF). 고지와 전문은 `licenses/` 폴더에 있고, 배포 zip에도 들어갑니다.
- 개발용 도구는 [Ultralytics](https://github.com/ultralytics/ultralytics)(AGPL-3.0)와 PyTorch를 씁니다(배포본에는 포함되지 않음). 모델이 Ultralytics로 학습되었으므로, 외부에 배포할 때는 이 점을 확인하세요.
