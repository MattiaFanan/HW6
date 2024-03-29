#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;
vector<Point2f> localize_object(const Mat& obj_image, Mat frame, const Mat& homography);
vector<DMatch> refine_match(const vector<DMatch>& matches ,const Mat& inliers_mask);
Mat get_homography(const vector<DMatch>& matches, const vector<KeyPoint>& obj_keypoints, const vector<KeyPoint>& frame_keypoints, Mat& inliers_mask);
void draw_boundaries(Mat frame,vector<Point2f> scene_corners, const Scalar& color);

int main(int argc, char *argv[]) {
    if (argc < 2){
        perror("Please provide data");
        return -1;
    }
    String path = argv[1];
    String obj_folder = "/objects/";
    vector<String> obj_file;
    glob(path+obj_folder+"obj*.png", obj_file);
    cout<<obj_file[0]<<endl;
    Scalar obj_color[] = {Scalar(255,0,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(255,0,255)};
    Scalar line_color = Scalar(0,255,0);
    int num_objects = static_cast<int>(obj_file.size());
    Mat obj_image[num_objects];

    Ptr<SIFT> siftPtr = SIFT::create();
    BFMatcher matcher(cv::NORM_L2,true);

    //read files
    for (int i=0; i < num_objects; i++)
        obj_image[i] = imread(obj_file[i]);

    //get keypoints and descriptors
    vector<vector<KeyPoint>> obj_keypoints(num_objects);
    vector<Mat> obj_descriptor(num_objects);
    for (int i=0; i<num_objects; i++)
        siftPtr->detectAndCompute(obj_image[i], noArray(), obj_keypoints[i], obj_descriptor[i]);


    VideoCapture cap(path+"/video.mov");
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    vector<Point2f> prevImgObjPoints[num_objects]; //vector of num_objects vectors of 2D points
    vector<Point2f> corners[num_objects];
    Mat prevImg;
    namedWindow("Frame", WINDOW_NORMAL);
    int cnt=0;
    while(true){
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        if (cnt==0) {
            cvtColor(frame, prevImg, COLOR_BGR2GRAY);

            vector<KeyPoint> frame_keypoints;
            Mat frame_descriptor;

            //compute obj_keypoints for frame
            siftPtr->detectAndCompute(frame, noArray(), frame_keypoints, frame_descriptor);

            //matches the keypoints
            vector<vector<DMatch>> matches(num_objects);
            for (int i=0; i<num_objects; i++)
                matcher.match(frame_descriptor, obj_descriptor[i], matches[i]);

            // matches refinement
            Mat inliers_mask[num_objects];
            Mat H[num_objects];
            vector<DMatch> good_matches[num_objects];
            for (int i=0; i<num_objects; i++) {
                H[i] = get_homography(matches[i], obj_keypoints[i], frame_keypoints, inliers_mask[i]);
                good_matches[i] = refine_match(matches[i], inliers_mask[i]);
            }

            //extract keypoint position of matches
            vector<KeyPoint> good_keypoint[num_objects];
            for (int i=0; i<num_objects; i++)
                for (DMatch match : good_matches[i]) {
                    KeyPoint matchedKeypoint = frame_keypoints.at(match.queryIdx);
                    good_keypoint[i].push_back(matchedKeypoint);
                    prevImgObjPoints[i].push_back(matchedKeypoint.pt);
                }

            //localize the object with a green quadrilateral
            for (int i=0; i<num_objects; i++) {
                corners[i] = localize_object(obj_image[i], frame, H[i]);
                draw_boundaries(frame, corners[i], line_color);
                drawKeypoints(frame, good_keypoint[i], frame, obj_color[i]);
            }
        }
        else{
            //Object tracking
            Mat frame_gray;
            cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
            vector<Point2f> currImgObjPoints[num_objects];
            vector<uchar> flowStatus_objKeypoints[num_objects];
            vector<float> err_objKeypoints[num_objects];
            for (int i = 0; i < num_objects; ++i) {
                calcOpticalFlowPyrLK(prevImg, frame_gray, prevImgObjPoints[i], currImgObjPoints[i],
                                     flowStatus_objKeypoints[i], err_objKeypoints[i]);
            }
            //Update prevImage with the new image
            copyTo(frame_gray, prevImg, Mat());

            //estimate optical flow

            Mat H_optflow[num_objects];
            for (int i = 0; i < num_objects; ++i) {
                //Compute the homography for the current movement
                H_optflow[i] = findHomography(prevImgObjPoints[i], currImgObjPoints[i], RANSAC,3, flowStatus_objKeypoints[i]);
                //Update corners position with the estimated movement
                perspectiveTransform(corners[i], corners[i], H_optflow[i]);
                //Draw new bounding boxes
                draw_boundaries(frame, corners[i], line_color);
            }

            //Discard points that can't be tracked
            for (int i = 0; i < num_objects; ++i) {
                prevImgObjPoints[i].clear();
                for (int j = 0; j < currImgObjPoints[i].size(); ++j)
                    if (flowStatus_objKeypoints[i][j] == 1)
                        prevImgObjPoints[i].emplace_back(currImgObjPoints[i][j]);
            }
        }

        imshow( "Frame", frame);
        cnt++;
        cout<<cnt<<endl;

        char c=(char)waitKey(1);
        if(c==27)
            break;
    }
    waitKey(0);
    cap.release();
    destroyAllWindows();
    return 0;
}

vector<Point2f> localize_object(const Mat& obj_image, Mat frame, const Mat& homography){
    //build object corners
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)obj_image.cols, 0 );
    obj_corners[2] = Point2f( (float)obj_image.cols, (float)obj_image.rows );
    obj_corners[3] = Point2f( 0, (float)obj_image.rows );

    //-- re-project the corners with the homography
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, homography);

    return scene_corners;
}

void draw_boundaries(Mat frame,vector<Point2f> scene_corners, const Scalar& color){
    //Draw lines
    line( frame, scene_corners[0],scene_corners[1], color, 4 );
    line( frame, scene_corners[1],scene_corners[2], color, 4 );
    line( frame, scene_corners[2],scene_corners[3], color, 4 );
    line( frame, scene_corners[3],scene_corners[0], color, 4 );
}

vector<DMatch> refine_match(const vector<DMatch>& matches ,const Mat& inliers_mask){
    //remove outliers
    vector<DMatch> good_matches;
    for (int i=0 ; i<matches.size(); i++)
        if (inliers_mask.at<bool>(0,i))
            good_matches.push_back(matches[i]);

    return good_matches;
}

Mat get_homography(const vector<DMatch>& matches, const vector<KeyPoint>& obj_keypoints, const vector<KeyPoint>& frame_keypoints, Mat& inliers_mask){

    vector<Point2f> obj_points;
    vector<Point2f> frame_points;
    // get centers of the keypoints
    for(DMatch match : matches)
    {
        obj_points.push_back(obj_keypoints[ match.trainIdx ].pt );
        frame_points.push_back( frame_keypoints[ match.queryIdx ].pt );
    }

    //get homography
    Mat H = findHomography( obj_points, frame_points, RANSAC,3, inliers_mask);
    return H;
}
