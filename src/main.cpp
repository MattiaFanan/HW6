#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;
void localize_object(const Mat& obj_image, Mat frame, const Mat& homography, const Scalar& color);
vector<DMatch> refine_match(const vector<DMatch>& matches ,const Mat& inliers_mask);
Mat get_homography(const vector<DMatch>& matches, const vector<KeyPoint>& obj_keypoints, const vector<KeyPoint>& frame_keypoints, Mat& inliers_mask);

int main(int argc, char *argv[]) {
    String path = argv[1];
    String obj_folder = "/objects";
    String obj_file[] = {"/obj1.png", "/obj2.png", "/obj3.png", "/obj4.png"};
    Scalar obj_color[] = {Scalar(255,0,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(255,0,255)};
    Scalar line_color = Scalar(0,255,0);
    int num_objects = sizeof(obj_file)/sizeof(obj_file[0]);
    Mat obj_image[num_objects];

    Ptr<SIFT> siftPtr = SIFT::create();
    BFMatcher matcher(cv::NORM_L2,true);

    //read files
    for (int i=0; i < num_objects; i++)
        obj_image[i] = imread(path + obj_folder + obj_file[i]);

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
                for (DMatch match : good_matches[i])
                    good_keypoint[i].push_back(frame_keypoints.at(match.queryIdx));

            //localize the object with a green quadrilateral
            for (int i=0; i<num_objects; i++) {
                localize_object(obj_image[i], frame, H[i], line_color);
                drawKeypoints(frame, good_keypoint[i], frame, obj_color[i]);
            }
        }

        imshow( "Frame", frame);
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

void localize_object(const Mat& obj_image, Mat frame, const Mat& homography, const Scalar& color){
    //build object corners
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)obj_image.cols, 0 );
    obj_corners[2] = Point2f( (float)obj_image.cols, (float)obj_image.rows );
    obj_corners[3] = Point2f( 0, (float)obj_image.rows );

    //-- re-project the corners with the homography
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, homography);

    //Draw lines
    line( frame, scene_corners[0],scene_corners[1], color, 4 );
    line( frame, scene_corners[1],scene_corners[2], color, 4 );
    line( frame, scene_corners[2],scene_corners[3], color, 4 );
    line( frame, scene_corners[3],scene_corners[0], color, 4 );
    return;
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
