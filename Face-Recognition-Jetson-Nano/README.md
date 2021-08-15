# Recognize faces with your Jetson Nano.
![output image]( https://qengineering.eu/images/John_Cleese.png )



------------

## Benchmark.
| Model  | Jetson Nano 2015 MHz | Jetson Nano 1479 MHz | RPi 4 64-OS 1950 MHz | RPi 4 64-OS 1500 MHz |
| ------------- | :------------: | :-------------: | :-------------:  | :-------------: |
| MTCNN  | 11 mS | 14 mS  | 22 mS | 25 mS  |
| RetinaFace  | 15 mS  | 19 mS  | 35 mS  | 37 mS  |
| ArcFace  | +17 mS | +21 mS  | +36 mS  | +40 mS  |
| Spoofing | +25 mS  | +37 mS  | +37 mS  | +45 mS  |


------------

## Dependencies.
### April 4 2021: Adapted for ncnn version 20210322 or later
To run the application, you have to:
- The Tencent ncnn framework installed. [Install ncnn](https://qengineering.eu/install-ncnn-on-jetson-nano.html) 
- Code::Blocks installed. (`$ sudo apt-get install codeblocks`)

------------


## Running the app.
To run the application load the project file FaceRecognition.cbp in Code::Blocks.<br/> 


------------

## Code.
```
#define RETINA                 
#define RECOGNIZE_FACE
#define TEST_LIVING
#define SHOW_LEGEND
#define SHOW_LANDMARKS
```
TEST_LIVING: By commenting the line the define is switched off. For instance, if you do not want to incorporate the **anti-spoofing** test (saves you 37 mS), 

------------


## WebCam.
If you want to use a camera 
`cv::VideoCapture cap(0);                          //WebCam`
If you want to run a movie
`cv::VideoCapture cap("Norton_2.mp4");   //Movie`

------------

## Papers.
[MTCNN](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)
[RetinaFace](https://arxiv.org/pdf/1905.00641.pdf)
[ArcFace](https://arxiv.org/pdf/1801.07698.pdf)
[Anti spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md)

------------

### Thanks.
https://github.com/Tencent/ncnn
https://github.com/nihui
https://github.com/LicheeX/MTCNN-NCNN
https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi
https://github.com/deepinsight/insightface
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing 
https://github.com/Qengineering/Blur-detection-with-FFT-in-C
