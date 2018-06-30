/* 加载单张图片并解码得到其data */

#pragma once

#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// Helper struct for loading a image
struct Image
{
    unsigned int m_label;
    std::vector<float> m_image;
};

// Load a single image
std::unique_ptr<Image> loadImage(std::string image_path, const int label)
{
    /* 此处使用OpenCV读取图片，也可以使用其它解码方法 */
	cv::Mat img = cv::imread(image_path);
	int totalpixel = 0;
	const unsigned int image_byte_size = img.channels() * img.rows * img.cols;
	std::vector<float> inputImageData;
	inputImageData.resize(image_byte_size);
	unsigned int countR_o = 0;
	unsigned int countG_o = img.rows * img.cols;
	unsigned int countB_o = 2 * img.rows * img.cols;
	unsigned int step = 1;
  	/* 逐像素提取data，注意OpenCV中颜色通道的排列顺序是BGR */
	for (int i = 0; i != img.rows; ++i)
	{
		for (int j = 0; j != img.cols; ++j)
		{
			inputImageData[countR_o] = static_cast<float>(img.data[totalpixel + 2]);
			inputImageData[countG_o] = static_cast<float>(img.data[totalpixel + 1]);
			inputImageData[countB_o] = static_cast<float>(img.data[totalpixel]);
			countR_o += step;
			countG_o += step;
			countB_o += step;
			totalpixel += 3;
		}
	}

    // store image and label in Image
    std::unique_ptr<Image> ret(new Image);
    ret->m_label = label;
    ret->m_image = std::move(inputImageData);

    return ret;
}
