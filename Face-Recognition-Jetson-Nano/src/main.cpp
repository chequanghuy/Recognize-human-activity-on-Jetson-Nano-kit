
#include <iostream>
#include <opencv2/opencv.hpp>
#include "TMtCNN.h"
#include "TArcface.h"
#include "TRetina.h"
#include "TWarp.h"
#include "TLive.h"
#include "TBlur.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include<thread>
//----------------------------------------------------------------------------------------
#define RETINA                  //comment if you want to use MtCNN landmark detection instead
#define RECOGNIZE_FACE
//#define TEST_LIVING
// some diagnostics
#define SHOW_LEGEND
#define SHOW_LANDMARKS
//----------------------------------------------------------------------------------------
// Adjustable Parameters
//----------------------------------------------------------------------------------------
const int   MaxItemsDatabase = 2000;
const int   MinHeightFace    = 90;
const float MinFaceThreshold = 0.50;
const float FaceLiving       = 0.93;
const double MaxBlur         = -25.0;   //more positive = sharper image
const double MaxAngle        = 10.0;
//----------------------------------------------------------------------------------------
// Some globals
//----------------------------------------------------------------------------------------
const int   RetinaWidth      = 320;
const int   RetinaHeight     = 240;
float ScaleX, ScaleY;
vector<cv::String> NameFaces;
//----------------------------------------------------------------------------------------
using namespace std;
using namespace cv;
using namespace cv::ml;
//----------------------------------------------------------------------------------------
//  Computing the cosine distance between input feature and ground truth feature
//----------------------------------------------------------------------------------------
inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2)
{
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}
//----------------------------------------------------------------------------------------
// painting
//----------------------------------------------------------------------------------------
void DrawObjects(cv::Mat &frame, vector<FaceObject> &Faces)
{
    for(size_t i=0; i < Faces.size(); i++){
        FaceObject& obj = Faces[i];

//----- rectangle around the face -------
        obj.rect.x *= ScaleX;
        obj.rect.y *= ScaleY;
        obj.rect.width *= ScaleX;
        obj.rect.height*= ScaleY;
        cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0));
//---------------------------------------

//----- diagnostic ----------------------
#ifdef SHOW_LANDMARKS
        for(int u=0;u<5;u++){
            obj.landmark[u].x*=ScaleX;
            obj.landmark[u].y*=ScaleY;
        }

        cv::circle(frame, obj.landmark[0], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[1], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[2], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[3], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[4], 2, cv::Scalar(0, 255, 255), -1);
#endif // SHOW_LANDMARKS
//---------------------------------------
#ifdef SHOW_LEGEND
        cv::putText(frame, cv::format("Angle : %0.1f", obj.Angle),cv::Point(10,40),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        cv::putText(frame, cv::format("Face prob : %0.4f", obj.FaceProb),cv::Point(10,60),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        cv::putText(frame, cv::format("Name prob : %0.4f", obj.NameProb),cv::Point(10,80),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
#ifdef TEST_LIVING
        if(obj.Color==2){
            //face is too tiny
            cv::putText(frame, cv::format("Live prob : ??"),cv::Point(10,100),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        }
        else{
            //face is ok
            cv::putText(frame, cv::format("Live prob : %0.4f", obj.LiveProb),cv::Point(10,100),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        }
#endif // TEST_LIVING
#endif // SHOW_LEGEND
//----- labels ----------------------------
#ifdef RECOGNIZE_FACE
        cv::String Str;
        cv::Scalar color;
        int  baseLine = 0;

        switch(obj.Color){
            case 0 : color = cv::Scalar(255, 255, 255); break;  //default white -> face ok
            case 1 : color = cv::Scalar( 80, 255, 255); break;  //yellow ->stranger
            case 2 : color = cv::Scalar(255, 237, 178); break;  //blue -> too tiny
            case 3 : color = cv::Scalar(127, 127, 255); break;  //red -> fake
            default: color = cv::Scalar(255, 255, 255);
        }

        switch(obj.NameIndex){
            case -1: Str="Stranger"; break;
            case -2: Str="too tiny"; break;
            case -3: Str="Fake !";   break;
            default: Str=NameFaces[obj.NameIndex];
        }

        cv::Size label_size = cv::getTextSize(Str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if(y<0) y = 0;
        if(x+label_size.width > frame.cols) x=frame.cols-label_size.width;

        cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),color, -1);
        cv::putText(frame, Str, cv::Point(x, y+label_size.height+2),cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0));
#endif // RECOGNIZE_FACE
    }
}
//----------------------------------------------------------------------------------------
// main
//----------------------------------------------------------------------------------------
void face(string name)
{
    float f;
    float FPS[16];
    int   n,Fcnt=0;
    size_t i;
    cv::Mat frame;
    cv::Mat result_cnn;
    cv::Mat faces;
    std::vector<FaceObject> Faces;
    vector<cv::Mat> fc1;
    string pattern_jpg = "./img/*.jpg";
    cv::String NewItemName;
    size_t FaceCnt;
    //the networks
    TLive Live;
    TWarp Warp;
    TMtCNN MtCNN;
    TArcFace ArcFace;
    TRetina Rtn(RetinaWidth, RetinaHeight, true);     //have Vulkan support on a Jetson Nano
    //some timing
    chrono::steady_clock::time_point Tbegin, Tend;

    Live.LoadModel();
    Ptr<SVM> svm = Algorithm::load<SVM>("trained-svm.xml");
    for(i=0;i<16;i++) FPS[i]=0.0;

    //OpenCV Version
    cerr << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "."
    << CV_SUBMINOR_VERSION << endl;
    cout << " " << endl;

#ifdef RECOGNIZE_FACE
    cout << "Trying to recognize faces" << endl;
    cout << " " << endl;
#ifdef RETINA
    cout << "Using Retina" << endl;
    cout << " " << endl;
#else
    cout << "Using MtCNN" << endl;
    cout << " " << endl;
#endif // RETINA

#ifdef TEST_LIVING
    cout << "Test living or fake fave" << endl;
    cout << " " << endl;
#endif // TEST_LIVING

#endif // RECOGNIZE_FACE



    // RaspiCam or Norton_2.mp4 ?
    // cv::VideoCapture cap(0);             //RaspiCam
    cv::VideoCapture cap(name);   //Movie
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return;
    }
    cout << "Start grabbing, press ESC on TLive window to terminate" << endl;

    while(1){
        Tbegin = chrono::steady_clock::now();
        cap >> frame;

        if (frame.empty()) {
            cerr << "End of movie" << endl;
            break;
        }
        ScaleX = ((float) frame.cols) / RetinaWidth;
        ScaleY = ((float) frame.rows) / RetinaHeight;

        // copy/resize image to result_cnn as input tensor
        cv::resize(frame, result_cnn, Size(RetinaWidth,RetinaHeight),INTER_LINEAR);



#ifdef RETINA
        auto startRetina = chrono::steady_clock::now();
        Rtn.detect_retinaface(result_cnn,Faces);
        auto endRetina = chrono::steady_clock::now();
#else
        MtCNN.detect(result_cnn,Faces);
#endif // RETINA

#ifdef RECOGNIZE_FACE
        //reset indicators
        for(i=0;i<Faces.size();i++){
            Faces[i].NameIndex = -2;    //-2 -> too tiny (may be negative to signal the drawing)
            Faces[i].Color     =  2;
            Faces[i].NameProb  = 0.0;
            Faces[i].LiveProb  = 0.0;
        }
        if (Faces.size()==1)
        for(i=0;i<Faces.size();i++){
            if(Faces[i].rect.height < MinHeightFace){
                Faces[i].Color = 2; //too tiny
            }
            else if(Faces[i].FaceProb>MinFaceThreshold){
                //get centre aligned image
                auto startForward = chrono::steady_clock::now();
                cv::Mat aligned = Warp.Process(result_cnn,Faces[i]);
                Faces[i].Angle  = Warp.Angle;
                //features of camera image
                cv::Mat fc2 = ArcFace.GetFeature(aligned);
                auto endForward = chrono::steady_clock::now();
                //fc2.convertTo(fc2,CV_32F);
                //reset indicators
                Faces[i].NameIndex = -1;    //a stranger
                Faces[i].Color     =  1;
                auto startSVM = chrono::steady_clock::now();
                float respone;
                respone=svm->predict(fc2);
                //cerr<<respone<<endl;
                auto endSVM = chrono::steady_clock::now();

#ifdef TEST_LIVING
                auto startLiv = chrono::steady_clock::now();
                                //test fake face
                float x1 = Faces[i].rect.x;
                float y1 = Faces[i].rect.y;
                float x2 = Faces[i].rect.width+x1;
                float y2 = Faces[i].rect.height+y1;
                struct LiveFaceBox LiveBox={x1,y1,x2,y2};

                Faces[i].LiveProb=Live.Detect(result_cnn,LiveBox);
                if(Faces[i].LiveProb<=FaceLiving){
                    Faces[i].Color     =  3; //fake
                    Faces[i].NameIndex = -3;
                }
                auto endLiv = chrono::steady_clock::now();
                std::cout << "Retina took " << std::chrono::duration_cast<chrono::milliseconds>(endRetina - startRetina).count() << "ms\n";
                std::cout << "Forward took " << std::chrono::duration_cast<chrono::milliseconds>(endForward - startForward).count() << "ms\n";
                std::cout << "SVM took " << std::chrono::duration_cast<chrono::milliseconds>(endSVM - startSVM).count() << "ms\n\n";
                std::cout << "Livtook " << std::chrono::duration_cast<chrono::milliseconds>(endLiv - startLiv).count() << "ms\n\n";
#endif // TEST_LIVING
            }
        }


#endif // RECOGNIZE_FACE

        //Tend = chrono::steady_clock::now();

        DrawObjects(frame, Faces);

//        calculate frame rate
        Tend = chrono::steady_clock::now();
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        cv::putText(frame, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));

        //show output

        cv::imshow(name, frame);


        //Faces.clear();
        char esc = cv::waitKey(5);
        if(esc == 27) break;


    }
    cv::destroyAllWindows();

    //return 0;
}
int main(){
    thread th1(face,"Norton_A.mp4");
    thread th2(face,"Norton_2.mp4");
    //face("Norton_2.mp4");
    return 0;
}

