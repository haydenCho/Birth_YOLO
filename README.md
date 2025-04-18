# Birth with YOLOv5
 **임산부를 위한 딥러닝 기반 수유실 자동 출입 시스템 'Birth'의 객체 인식 모델(YOLO) 리포지토리**

<br/>

## 프로젝트 논문
- [영상분석 기반 임산부 인증 수유실 자동 출입 관리](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11990080)
    - DOI: 10.6109/jkiice.2024.28.11.1379

<br/>

## 프로젝트 개요
### 1. 프로젝트 주제 및 목표
**이 프로젝트는 지하철 내 수유실의 보안성과 사용 편의성을 동시에 향상시키기 위해 영상 분석 기반의 수유실 출입 자동 인식 시스템을 구현하는 것을 목표로 한다.** <br/>
현재 대부분의 수유실은 출입 통제가 없어 임산부들이 불편함과 불안감을 겪고 있으며, 일부 지하철역에서는 역무원이 직접 문을 열어주는 방식으로 운영되어 인력 낭비 및 사용자 불편이 발생한다. 수유실 자동 출입 시스템은 카메라를 통해 사용자의 존재를 확인하고, 딥러닝 기반 객체 인식 모델인 YOLOv5와 Mediapipe를 활용해 수유 대상자인지를 자동으로 판단함으로써, 별도의 인력 개입 없이 자동문 제어가 가능하도록 한다. 이를 통해 수유실 이용의 안정성과 효율성을 동시에 확보하고, 임산부의 이동 편의성을 실질적으로 개선하는 것을 주요 목표로 한다.

<br/>

### 2. 개발 기간
- 전체 기간: 2023.10.14 ~ 2024.06.14 (약 8개월)
    - 데이터셋 구축: 2023.10.14 ~ 2024.01.26
    - 모델 학습: 2024.01.27 ~ 2024.06.14

<br/>

### 3. 개발 환경
- 로컬 컴퓨터 GPU: Nvidia RTX 3060
- CUDA 버전: 11.8
- 사용 도구: Visual Studio Code, Colab(Google)

<br/>

### 4. 데이터셋
- [Baby Finder Dataset](https://universe.roboflow.com/haydencho/baby-finder/dataset/7) <br/>
    <img src="https://github.com/user-attachments/assets/d76ebdc7-9cdf-4c02-83c4-8ac6ee168053" width="300"/>
- 모델 학습을 위해 데이터셋 생성 및 관리를 지원하는 Roboflow 서비스를 활용하여 이미지를 수집해서 바운딩 박스를 생성하고 전처리하여 데이터셋 구축한다.
- 데이터셋의 이미지 클래스는 아기(baby), 아이(kids), 성인(adult), 유아차(stroller)로, 주요 클래스는 아기(baby)와 유아차(stroller)이다.
- 클래스당 데이터 개수(클래스가 포함된 이미지 개수)
    - adult: 1,154
    - kids: 988
    - baby: 943
    - stroller: 469
- 데이터셋 이미지 개수: 3896
    - train: 3675
    - test: 107
    - valid: 114

<br/>

#### 데이터셋 구축 과정
1. Roboflow 플랫폼에서 나이대별 인물 이미지 데이터와 유아차 이미지 데이터를 수집<br/>
2. 필요한 클래스 라벨을 생성하고 이미지에 존재하는 객체에 해당하는 라벨로 바운딩 박스(bounding Box)를 생성<br/>
    - 라벨링을 한 결과, 아기(baby) 클래스의 데이터가 지나치게 적은 문제점을 발견하여 이를 해결하고자 웹 크롤링을 통해 데이터를 추가 수집하여 클래스 불균형을 해결
3. 바운딩 박스를 생성한 이미지에 대해 크기 조절, 각도 회전, 그레이 스케일 등의 전처리 및 추가 작업을 진행<br/>
    - 이미지의 사이즈: YOLO 모델에 적합한 640x640
    - 이미지 각도: ±12° (지하철 환경 고려)
    - 노이즈 픽셀 추가 작업 (지하철 환경 고려)

<br/>

### 4. 기대되는 결과 및 기여
이 프로젝트를 통해 구현되는 자동 인식 기반 수유실 출입 시스템은 다음과 같은 긍정적인 효과를 기대할 수 있다.

1. **임산부와 보호자의 수유실 이용 편의성 향상**: 자동 출입 시스템을 통해 불필요한 대기나 역무원 호출 없이 수유실 이용이 가능하게 한다.
2. **지하철 운영의 효율성 증진**: 역무원의 업무 부담을 줄이고, 인력 자원의 효율적 배치를 가능하게 한다.
3. **보안성 강화**: 수유실이 필요한 사용자인지를 자동으로 인식함으로써 외부인의 무단 출입을 차단할 수 있어 사용자들의 불안감을 해소할 수 있다.
4. **기술적 확장 가능성**: 수유실 자동 출입 시스템은 추후 다른 대상(노인, 장애인 등)이나 다른 공공장소(공항, 대형 쇼핑몰 등)에도 적용 가능하여, 노약자의 공공 공간 접근성을 높이는 데 기여할 수 있다.

<br/>

## 버전 선택 이유
| Version | Feature |
|  -----  | ------- |
|    5    | - 경량화 모델을 지원<br/>- 규모별로 n, s, m, l, xl 모델 제공    |
|    6    | - 포즈 추정 기능 추가    |
|    7    | - v4 기반 확장    |
|    8    | - 감지, 포즈 추정, 추적기능 고도화    |
|    9    | - PGI와 GELAN를 도입하여 정보손실 문제를 해결    |
- 개발 당시 YOLO 모델은 v1부터 v9까지 9개의 모델이 발표되었다. 그중 비교적 최근에 발표되었으며 준수한 성능을 가진 모델로 평가되는 v5부터 v9까지의 모델 특징을 위와 같이 비교하여 모델을 선택한다.
- 성능만 고려한다면 v8이나 v9을 선택하는 것이 합리적이며 성능과 속도가 모두 중요하다면 v5를 사용하는 것이 합리적이라고 판단하였다.
- **수유실 자동 출입 시스템에서는 모델이 실시간으로 객체를 인식해야 하므로 속도가 중요하며 라즈베리파이에서 모델을 사용하여 모델의 크기도 고려해야 했기 때문에 v5를 선택하여 집중적으로 학습한다.**

<br/>

## 모델 학습
#### 모델 학습 코드
- [공식 깃허브](https://github.com/ultralytics/yolov5)
- [공식 문서](https://docs.ultralytics.com/ko/models/yolov5/#performance-metrics)
- 학습 코드는 공식 깃허브의 코드를 로컬로 받아 사용
    ```
  python train.py --batch 16 --epochs 30 --data C:/Birth/myvenv/yolov5/dataset/data.yaml --hyp data/hyps/hyp_evolve.yaml --name birth_test_xl_5_3
    ```

<br/>

### YOLOv5 모델
- YOLOv5 모델은 모델 규모에 따라 n, s, m, l, x 모델로 나뉜다.
    - 모델 크기가 작은 n 모델은 성능이 떨어지지만 가벼워서 실시간 객체 인식이 가능하다는 장점이 있고, 모델 크기가 큰 x 모델은 상대적으로 느리지만 객체 인식 성능이 뛰어나다는 장점이 있다.
    - 이 프로젝트에서는 처리 속도도 중요하지만, 실시간에 견줄만한 속도가 필요하진 않으며 보다 객체 인식 성능이 중요하다는 것을 고려하여 n 모델을 제외한 **s, m, l, x 모델을 학습**한다.<br/>

| Model-size | 정밀도 (Precision) | 재현율 (Recall) | mAP_0.5 |
|  --------  | ------------------ | --------------- | -------- |
|  YOLOv5-s  | 66.6               | 67.2            | 69.6     |
|  YOLOv5-m  | 85.7               | 79.8            | 87.0     |
|  YOLOv5-l  | 86.6               | 79.6            | 87.2     |
|  YOLOv5-x  | 87.8               | 85.6            | 90.4     |

#### 모델 크기 선택
- 학습 결과, x 모델이 다른 모델에 비해 좋은 성능을 보였으며 특히 정밀도와 재현율 두 지표가 균형을 이루는 점에서 유의미한 결과라고 판단하여 **x 모델을 추가 학습**하는 것으로 결정하였다.
- *x 모델 추가 학습은 학습률을 변경하며 최적의 하이퍼 파라미터를 찾는 방향으로 진행*하였으며 결과는 아래 성능 평가 부분과 같다.

<br/>

### 학습 파라미터
- 버전 선택 초기 파라미터<br/>

    | img-size | batch | epoch | lr0 | lrf |
    |  ------  | ----- | ----- | ---- | --- |
    |  640x640 |  16   |  30   | 0.01 | 0.1 |

<br/>

## 모델 평가
### 성능 평가 지표
- 모델 성능을 평가하기 위해 *정밀도(Precision), 재현율(Recall), 그리고 mAP(Mean Average Precision)* 지표를 사용
    - 정밀도(Precision): 모델이 긍정(Positive)으로 예측한 결과 중 실제로 긍정인 비율
    - 재현율(Recall): 실제로 긍정인 것 중 모델이 정확히 긍정으로 예측한 비율
    - AP(Average Precision): 정밀도-재현율(Precision-Recall) 곡선 아래의 면적을 계산한 값, 객체 검출 모델의 성능을 종합적으로 평가하는 지표이며, mAP은 AP의 평균
- **이 프로젝트에서는 사용자 안전을 위해 오검출을 최소화하는 것이 중요하며, 편리성을 위해 미검출도 줄여야 한다. 따라서 최종적으로 모델 성능을 비교할 때는 정밀도와 재현율을 모두 반영한 mAP 지표를 기준으로 평가한다.**

<br/>

## 모델 경량화 및 가속화
- 라즈베리파이에서 모델을 사용하기 위해 다음과 같이 변환 시도
- 공식 깃허브 제공 코드 중 export.py 사용하여 변환 <br/>

### TensorFlow Lite(.tflite)
- 모바일에서 많이 사용
- 가장 기본적인 경량화 방법

### ONNX(Open Neural Network Exchange, .onnx)
- 기계 학습 모델을 표현하기 위해 만들어진 오픈 포맷
- CPU에서 사용할 때 속도 증가

<br/>

#### 결과
- 경량화와 가속화 모두 효과가 있었으나 실질적으로 전반적인 서비스에 도움이 된 것은 가속화(ONNX)였다.
- ONNX 런타임 등을 적용하면 더 효과가 있을 것 같다는 피드백

<br/>

### 평가 결과
| Num | lr0 | lrf | 정밀도 (Precision) | 재현율 (Recall) | mAP_0.5 |
|  -  | --- | --- | ----------------- | --------------- | --------------- |
|  1  | 0.01 | 0.1 | 87.8             | 85.6            | 90.4            |
|  2  | 0.1 | 0.1 | 68.6             | 79.6            | 87.2            |
|  3  | 0.001 | 0.01 | 85.9          | 87.5            | 91.9            |


<br/>

## Reference
- [공식 깃허브](https://github.com/ultralytics/yolov5)
- [공식 문서](https://docs.ultralytics.com/ko/models/yolov5/#performance-metrics)
- http://www.civicnews.com/news/articleView.html?idxno=14821
- J.H Choi, and K.S Nam, "Lactation Room Design Guidelines for Birth Friendly Environment," Journal of the Korean Society Design Culture, vol. 20, no. 4, pp. 721-730, Dec. 2014.  UCI : G704-001533.2014.20.4.040
- K.M. Hosny, N.A. Ibrahim, E.R. Mohamed, and H.M. Hamza, "Artificial intelligence-based masked face detection: A survey," Intelligent Systems with Applications, vol. 22, 200391, Jun. 2024. DOI: 10.1016/j.iswa.2024.200391
- K. V. Nuthan, B. V. S. Krishna, R. S. Gound, “Survey on Face Recognition using an improved VGGNet Convolutional Neural Network,” in Proceeding of International Conference on Sustainable Computing and Smart Systems, Coimbatore, India, 2023, DOI : 10.1109/ICSCSS57650.2023.10169193
- J.R. Lee, K. W. Ng, and Y. J. Yoong, “Face and Facial Expressions Recognition System for Blind People Using ResNet50 Architecture and CNN,” Journal of Information and Web Engineeing, vol. 2, no. 2, Sep. 2023. DOI :  10.33093/jiwe.2023.2.2.20
- H.W. Kim, B.Y. Ko, J.H Shim, W.Y. Chung, and E.J. Hwang, "A Front Face Recognition Scheme Using FaceNet and Facial Landmark Points Detector," in Proceedings of Korean Institute of Information Scientists and Engineers, Seoul, Korea, pp. 364-366, Dec. 2020.
- S. Jeon, J. Lee, M. Kim, S. Hong, J. Bang, and H. Kim, "A Study on Helmet Wear Identification Using YOLOv5 Object Recognition Model," in Proceedings of Symposium of the Korean Institute of Communications and Information Sciences, pp. 1293-1294, Pyeongchang, Korea, Feb. 2022. 
- S.Y. Jeong, M.J. Kang, and B.Y. Lee, "YOLOv5-based Fall Detection Research for Elderly Living Alone," JOURNAL OF THE KOREA CONTENTS ASSOCIATION, vol. 23, no. 11, pp. 83-89, Nov. 2023. DOI: https://doi.org/10.5392/ JKCA.2023.23.11.083
- I.C. Choi, T.H. Kim, H.S. Seo, and J.H. Ock, "Development of Automated Access Mangement Solution through CCTV Face Recognition based on Artificail Intelligence," in Proceedings of A collection of papers for the Korean Architectural Association's academic presentation, Busan, Korea pp. 1139-1142, Apr. 2023. Available: https:// www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE11427576
- J.T. Kim, M.H. Lee, H.K. Lee, S.H. Lee, and W.G. Lee, "Design of a bus-entrance doors tracking and guiding system for navigating visually impaired people via a modified YOLO algorithm," in Proceedings of Institute of Control, Robotics and Systems, Gangwon, Korea, pp. 242-243, Jul. 2020. 
- H.D. Ghael, L. Solanki, and G. Sahu, "A Review Paper on Raspberry Pi and its Applications," International Journal of Advances in Engineering and Management, vol. 2, no. 12, pp. 225-227, Dec. 2020. DOI: 10.35629/5252-0212225227
- MediaPipe, Retrieved May. 6, 2024. Available:https://ai.google.dev/edge/mediapipe/solutions/guide
- J.Y. Kim, "A comparative study on the characteristics of each version of object detection model YOLO," in Proceedings of Proceedings of the Korean Society of Computer Information Conference , Daejeon, Korea, pp. 75-78, Jul. 2023. Available: https://www.dbpia.co.kr/pdf/ pdfView.do?nodeId=NODE11528162
- H.J. Gwak, Y.J. Jeong, I.J. Chun, and C.H. Lee, "Estimation of fruit number of apple tree based on YOLOv5 and regression model," Journal of IKEEE, vol. 28, no. 2, pp. 28-35, Jun. 2024. Available: https://www.dbpia.co.kr/pdf/ pdfView.do?nodeId=NODE11840256
-  J.H. Kang, “Distance-based Adaptive Anchor box Selection for Object Detection and Localization with Magnetic Declination Correction in Drone,” Journal of Institute of Control, Robotics and Systems,  vol. 27, no. 10, pp. 776-783, Oct. 2021. DOI : 10.5302/J.ICROS.2021.21.0092
- H.J. Kim, “Breastfeeding Room: Moms Only”,  gyeong-sang-il-bo, Apr. 2019.  Available: https://www.ksilbo.co.kr/news/articleView.html?idxno=691443
