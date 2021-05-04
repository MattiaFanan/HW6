#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    String path = argv[1];
    String obj_folder = "/objects";
    String obj_file[] = {"/obj1.png", "/obj2.png", "/obj3.png", "/obj4.png"};
    //cout << path + obj_folder + obj_file[0];
    Mat obj1 = imread(path + obj_folder + obj_file[0]);

    Ptr<SIFT> siftPtr = SIFT::create();
    BFMatcher matcher(cv::NORM_L2,true);

    vector<KeyPoint> keypoints;
    Mat obj_descriptor;
    siftPtr->detect(obj1, keypoints, obj_descriptor);

    // Add results to image.
    Mat output;
    drawKeypoints(obj1, keypoints, output);


    VideoCapture cap(path+"/video.mov");
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    int cnt=0;
    while(true){
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        Mat imageMatches;
        if (cnt==0) {

            vector<KeyPoint> frame_keypoints;
            Mat frame_descriptor;
            siftPtr->detect(frame, frame_keypoints, frame_descriptor);

            vector<DMatch> matches;
            matcher.match(frame_descriptor, obj_descriptor, matches);
            cout << "match=";
//the number of matched features between the two images
            cout << matches.size() << endl;
            drawMatches(frame, frame_keypoints, obj1, keypoints, matches, imageMatches);
        }
        namedWindow("matches");
        imshow("matches",imageMatches);
        //imshow( "Frame", frame );
        cnt++;
        char c=(char)waitKey(25);
        if(c==27)
            break;
        break;
    }
    waitKey(0);
    cap.release();
    destroyAllWindows();
    return 0;
}
