#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <thread>
#include "Orient2D.hpp"
#include <dirent.h>
#include <sys/stat.h>

void save_orient_image(std::string orient_image_fn,  std::string out_orient_image_fn)
{
    cv::Mat orient_img = cv::imread(orient_image_fn, cv::IMREAD_UNCHANGED);
    cv::Mat orient_png(orient_img.rows, orient_img.cols, CV_8UC3);
    cv::Mat orient_exr(orient_img.rows, orient_img.cols, CV_32FC3);
    
    /* added by chenghong, save .png format orientation map */
    // orient_exr=orient_img*255;
    // orient_exr.convertTo(orient_png, CV_8UC3);
    // cv::imwrite(out_orient_image_fn, orient_png);
    // /* added by chenghong, remove float orient exr image */ 
    // remove((out_orient_image_fn + ".exr").c_str());
}

void save_orient_image_from_folder(std::string input_image_folder, std::string out_orient_image_folder)
{
    DIR *dirp;
    struct dirent *directory;

    dirp = opendir(input_image_folder.data());

    if (!dirp)
    {
        std::cout << "No input folder!!" << std::endl;
        exit(1);
    }

    while((directory = readdir(dirp)) != NULL )
    {
        std::string dir_name = directory->d_name;
        if (dir_name.length() < 3)
            continue;
        if ((dir_name.substr(dir_name.size()-4, dir_name.size()) != ".png") && (dir_name.substr(dir_name.size()-4, dir_name.size()) != ".jpg"))
            continue;
        
        std::string input_img_fn = input_image_folder + dir_name;
        std::string eyebrow_name = dir_name.substr(0, dir_name.size()-4);
        std::cout << "Process " << eyebrow_name << std::endl;

        std::string orient_img_fn = out_orient_image_folder + eyebrow_name + ".exr";
        std::string out_img_fn = out_orient_image_folder + eyebrow_name;

        std::ifstream infile(orient_img_fn);
        if(!infile.good())
            COrient2D orient2D (input_img_fn.data(),orient_img_fn.data());   //

        std::cout << "Finish processing " << eyebrow_name << std::endl;
    }
}

int main(int argc, char** argv)
{
    cv::String keys =          
        "{@input_eyebrow_img_fn    |           | Input eyebrow image}"           // input image is the first argument (positional)
        "{@orient_img_fn           |           | Output orient image}"    // optional, default value ""
        "{help                     |           | show help message}";             // optional, show help optional


    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string input_eyebrow_img_fn = parser.get<cv::String>(0);    // read @origin_images_dir (mandatory, error if not present)
    std::cout << "Input eyebrow image: " << input_eyebrow_img_fn.c_str() << std::endl;  
    std::string orient_img_fn = parser.get<cv::String>(1);


    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    // mkdir(out_orient_images_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    // save_orient_image_from_folder(input_orig_images_dir.c_str(),  out_orient_images_dir.c_str());

    std::ifstream infile(orient_img_fn);
    if(!infile.good())
        COrient2D orient2D (input_eyebrow_img_fn.data(),orient_img_fn.data());   //
     
    save_orient_image(orient_img_fn, orient_img_fn);
}