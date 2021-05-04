#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    String path = argv[1];
    String obj_folder = "/objects";
    String obj_file[] = {"/obj1.png", "/obj2.png", "/obj3.png", "/obj4.png"};
    //cout << path + obj_folder + obj_file[0];
    Mat obj_image = imread(path + obj_folder + obj_file[0]);

    Ptr<SIFT> siftPtr = SIFT::create();
    BFMatcher matcher(cv::NORM_L2,true);

    vector<KeyPoint> obj_keypoints;
    Mat obj_descriptor;
    siftPtr->detectAndCompute(obj_image, noArray(), obj_keypoints, obj_descriptor);


    VideoCapture cap(path+"/video.mov");
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    namedWindow("Frame", WINDOW_NORMAL);

    int cnt=0;
    while(true){
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        if (cnt==0) {

            vector<KeyPoint> frame_keypoints;
            Mat frame_descriptor;

            //compute obj_keypoints for frame
            siftPtr->detectAndCompute(frame, noArray(), frame_keypoints, frame_descriptor);

            //matches the obj_keypoints
            vector<DMatch> matches;
            matcher.match(frame_descriptor, obj_descriptor, matches);

            //RSAC
            vector<Point2f> obj_points;
            vector<Point2f> frame_points;
            // get centers of the keypoints
            for( size_t i = 0; i < matches.size(); i++ )
            {
                obj_points.push_back(obj_keypoints[ matches[i].trainIdx ].pt );
                frame_points.push_back( frame_keypoints[ matches[i].queryIdx ].pt );
            }

            //get homography
            Mat H = findHomography( obj_points, frame_points, RANSAC);

            //-- re-project the corners of the train object into the frame with the found homography
            std::vector<Point2f> obj_corners(4);
            obj_corners[0] = Point2f(0, 0);
            obj_corners[1] = Point2f( (float)obj_image.cols, 0 );
            obj_corners[2] = Point2f( (float)obj_image.cols, (float)obj_image.rows );
            obj_corners[3] = Point2f( 0, (float)obj_image.rows );

            std::vector<Point2f> scene_corners(4);
            perspectiveTransform( obj_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            line( frame, scene_corners[0],scene_corners[1], Scalar(0, 255, 0), 4 );
            line( frame, scene_corners[1],scene_corners[2], Scalar( 0, 255, 0), 4 );
            line( frame, scene_corners[2],scene_corners[3], Scalar( 0, 255, 0), 4 );
            line( frame, scene_corners[3],scene_corners[0], Scalar( 0, 255, 0), 4 );

            //extract keypoint of matches
            vector<KeyPoint> kp_match;
            for (DMatch match : matches)
                kp_match.push_back(frame_keypoints.at(match.queryIdx));

            //draw matches on frame
            drawKeypoints(frame, kp_match, frame, Scalar(255,0,0));
        }

        imshow( "Frame", frame );
        cnt++;

        char c=(char)waitKey(25);
        if(c==27)
            break;
        //TODO remove stub break
        break;
    }
    waitKey(0);
    cap.release();
    destroyAllWindows();
    return 0;
}
