#pragma once
#include "opencv2/highgui.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <stdio.h>
#include "DS_timer.h"
#include "DS_definitions.h"
#include <omp.h>

using namespace cv;
using namespace std;

namespace pre {
    void equalizeHistBGR_Serial(Mat& src, Mat& dst);

    void equalizeHistUsingHSL_Serial(Mat& src, Mat& dst);

    void split_Serial(Mat& input, std::vector<Mat>& channels);

    void CustomcalcHist_Serial(Mat& src, vector<int>& hist, int thread);


    void equalizeHistUsingBGR_Parallel(Mat& src, Mat& dst);

    void split_Parallel(Mat& input, std::vector<Mat>& channels);

    void equalizeHistUsingHSL_Parallel(Mat& src, Mat& dst);

    void equalizeHistBGR_Parallel(Mat& src, Mat& dst);

    void CustomcalcHist_Parallel(Mat& src, vector<int>& hist, int thread);

    void CustomaddWeighted_Serial(const Mat& d1, double alpha, const Mat& d2, double beta, Mat& dst);

    void CustomaddWeighted_Parallel(const Mat& d1, double alpha, const Mat& d2, double beta, Mat& dst);


    Mat merge_Serial(Mat& R, Mat& G, Mat& B);
    Mat merge_Parallel(Mat& R, Mat& G, Mat& B);

    int can_div_threads = 3;

    int histogramEqualizationSerial(Mat& input_image, Mat& output_image) {

        Mat rgb_eq, hls_eq;
        Mat dst_serial;
        Mat dst2_serial;
        Mat dst_parallel;
        Mat dst2_parallel;
        Mat result_serial;
        Mat result_parallel;
        vector<Mat> bgr_planes;
        vector<Mat> bgr_planes_results(3);
        vector<Mat> hls_planes;
        //output_image.copySize(input_image);


        if (omp_get_max_threads() < can_div_threads * 4) {
            can_div_threads = omp_get_max_threads() / 8;
        }

        if (input_image.empty())
        {
            printf("Image is Empty");
            return -1;
        }


        /* ============================ Serial Algorithm ======================*/
        /*---------------------------------BGR---------------------------------*/

        //OpenCV BGR���� B, G, R�� �����Ͽ� ����
        split_Serial(input_image, bgr_planes);

        //BGR���� ���Ͽ� ������׷� ��źȭ ����
        for (int i = 0; i < 3; i++)
        {
            equalizeHistBGR_Serial(bgr_planes[i], bgr_planes_results[i]);
        }

        //BGR���� �ٽ� �ϳ��� �̹����� ��ġ��
        dst_serial = merge_Serial(bgr_planes_results[2], bgr_planes_results[1], bgr_planes_results[0]);
        /*-------------------------------------------------------------------*/

        output_image = dst_serial;
    }


    void HistogramEqualizationOpenMP(Mat& input_image, Mat& output_image) {
        Mat rgb_eq, hls_eq;
        Mat dst_parallel;
        Mat dst2_parallel;
        Mat result_parallel;
        /*========================== Parallel Algorithm =========================*/

        //���� ��ø ���
        omp_set_nested(1);

        //section���� HSL�� BGR�� ó���� ���������� �����ϵ��� ��.
#pragma omp parallel
        {
#pragma omp sections
            {
#pragma omp section
                {
                    equalizeHistUsingBGR_Parallel(input_image, dst_parallel);
                }
            }

        }
        output_image = dst_parallel;
        //output_image.data = result_parallel.data;
        /*=========================Parallel Algorithm End =======================*/

    }

    //BGR���� ���� ������׷� ��Ȱȭ �Լ� ����
    void equalizeHistUsingBGR_Parallel(Mat& src, Mat& dst) {
        vector<Mat> bgr_planes;
        vector<Mat> bgr_planes_results(3);

        split_Parallel(src, bgr_planes);

        //RGB ������ �迭�� ���ؼ� �����ϹǷ� Thread�� ���� 3
#pragma omp parallel for num_threads(3)
        for (int i = 0; i < 3; i++)
        {
            //BGR���� ���Ͽ� ������׷� ��źȭ ����
            equalizeHistBGR_Parallel(bgr_planes[i], bgr_planes_results[i]);
        }
        dst = merge_Parallel(bgr_planes_results[2], bgr_planes_results[1], bgr_planes_results[0]);
    }


    //BGR������׷� �⺻ ���ȭ �Լ�
    void equalizeHistBGR_Serial(Mat& src, Mat& dst) {
        //������׷� ���
        vector<int> hist;
        CustomcalcHist_Serial(src, hist, omp_get_thread_num());

        //������׷� ����ȭ
        vector<int> cdf(256, 0);
        cdf[0] = hist[0];
        int totalPixels = src.rows * src.cols;
        vector<int> cdfNormalized(256, 0);
        cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
            cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
            //printf("%d \n", cdfNormalized[i]);
        }

        dst = Mat(src.rows, src.cols, src.type());
        //������׷� ��Ȱȭ
        for (int row = 0; row < src.rows; ++row) {
            const uchar* srcPtr = src.ptr<uchar>(row);
            uchar* dstPtr = dst.ptr<uchar>(row); // dstPtr Ÿ�� ����
            for (int col = 0; col < src.cols; ++col) {
                dstPtr[col] = static_cast<uchar>(cdfNormalized[srcPtr[col]]); // �ȼ� ���� ��Ȱȭ�� ������ ����
            }
        }
    }

    //BGR������׷� ���� ���ȭ �Լ�
    void equalizeHistBGR_Parallel(Mat& src, Mat& dst) {
        // ������׷� ���
        vector<int> hist;
        CustomcalcHist_Parallel(src, hist, omp_get_thread_num());

        vector<int> cdf(256, 0);
        cdf[0] = hist[0];
        int totalPixels = src.rows * src.cols;
        vector<int> cdfNormalized(256, 0);

        //������׷� ����ȭ
        cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
            cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
        }

        dst = Mat(src.rows, src.cols, src.type());
        //������׷� ��Ȱȭ
#pragma omp parallel for num_threads(can_div_threads)
        for (int row = 0; row < src.rows; ++row) {
            const uchar* srcPtr = src.ptr<uchar>(row);
            uchar* dstPtr = dst.ptr<uchar>(row); // dstPtr Ÿ�� ����
            for (int col = 0; col < src.cols; ++col) {
                dstPtr[col] = static_cast<uchar>(cdfNormalized[srcPtr[col]]); // �ȼ� ���� ��Ȱȭ�� ������ ����
            }
        }
    }

    //���� �⺻ �Լ�
    void split_Serial(Mat& input, std::vector<Mat>& channels) {
        int rows = input.rows;
        int cols = input.cols;

        channels.resize(3);
        channels[0] = Mat(rows, cols, CV_8UC1);
        channels[1] = Mat(rows, cols, CV_8UC1);
        channels[2] = Mat(rows, cols, CV_8UC1);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                uchar* pixel = input.ptr<uchar>(i, j);
                channels[0].at<uchar>(i, j) = pixel[0];
                channels[1].at<uchar>(i, j) = pixel[1];
                channels[2].at<uchar>(i, j) = pixel[2];
            }
        }
    }

    //���� ���� �Լ�
    void split_Parallel(Mat& input, std::vector<Mat>& channels) {
        int rows = input.rows;
        int cols = input.cols;

        channels.resize(3);
        channels[0] = Mat(rows, cols, CV_8UC1);
        channels[1] = Mat(rows, cols, CV_8UC1);
        channels[2] = Mat(rows, cols, CV_8UC1);


#pragma omp parallel for num_threads(can_div_threads*2)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                uchar* pixel = input.ptr<uchar>(i, j);
                channels[0].at<uchar>(i, j) = pixel[0];
                channels[1].at<uchar>(i, j) = pixel[1];
                channels[2].at<uchar>(i, j) = pixel[2];
            }
        }
    }


    //merge �ø��� �Լ�
    Mat merge_Serial(Mat& R, Mat& G, Mat& B) {
        Mat merged(R.rows, R.cols, CV_8UC3);
        for (int i = 0; i < R.rows; ++i) {
            for (int j = 0; j < R.cols; ++j) {
                merged.at<Vec3b>(i, j)[0] = B.at<uchar>(i, j);
                merged.at<Vec3b>(i, j)[1] = G.at<uchar>(i, j);
                merged.at<Vec3b>(i, j)[2] = R.at<uchar>(i, j);
            }
        }
        return merged;
    }

    //merge �����Լ�
    Mat merge_Parallel(Mat& R, Mat& G, Mat& B) {
        Mat merged(R.rows, R.cols, CV_8UC3);

#pragma omp parallel for num_threads(can_div_threads*2)
        for (int i = 0; i < R.rows; ++i) {
            for (int j = 0; j < R.cols; ++j) {
                merged.at<Vec3b>(i, j)[0] = B.at<uchar>(i, j);
                merged.at<Vec3b>(i, j)[1] = G.at<uchar>(i, j);
                merged.at<Vec3b>(i, j)[2] = R.at<uchar>(i, j);
            }
        }
        return merged;
    }

    //������׷� �ľ� �⺻ �Լ�
    void CustomcalcHist_Serial(Mat& src, vector<int>& hist, int thread = -1) {
        hist = vector<int>(256, 0);
        for (int row = 0; row < src.rows; ++row) {
            unsigned char* ptr = src.ptr<uchar>(row);
            for (int col = 0; col < src.cols; ++col) {
                hist[ptr[col]]++;
            }
        }
    }

    //������׷� �ľ� ���� �Լ�
    void CustomcalcHist_Parallel(Mat& src, vector<int>& hist, int thread = -1) {
        hist = vector<int>(256, 0);
#pragma omp parallel num_threads(can_div_threads*2)
        {
            /* no wait�� ����Ͽ� �ڽ��� ������ ���� ���� �ٷ� ������� ���Եǵ��� ��*/
            vector<int> tmp(256, 0);
#pragma omp for nowait
            for (int row = 0; row < src.rows; ++row) {
                unsigned char* ptr = src.ptr<uchar>(row);
                for (int col = 0; col < src.cols; ++col) {
                    tmp[ptr[col]]++;
                }
            }

            for (int i = 0; i < 256; i++) {
#pragma omp atomic
                hist[i] += tmp[i];
            }
        }
    }


    //HSL�⺻ó�� �Լ�
    void equalizeHistUsingHSL_Serial(Mat& src, Mat& dst) {
        Mat hsl;
        vector<Mat> channels;
        vector<int> hist;

        // BGR���� HSL�� ��ȯ
        cvtColor(src, hsl, COLOR_BGR2HLS);

        // HSLä�� �迭�� ���� ������
        split_Serial(hsl, channels);

        // L ������׷� Ȯ��
        CustomcalcHist_Serial(channels[1], hist);

        // ������׷� ����ȭ
        vector<int> cdf(256, 0);
        cdf[0] = hist[0];
        int totalPixels = channels[1].rows * channels[1].cols;
        vector<int> cdfNormalized(256, 0);
        cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
            cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
        }

        // L������׷� ��Ȱȭ
        dst = Mat(channels[1].rows, channels[1].cols, CV_8UC1);
        for (int row = 0; row < channels[1].rows; ++row) {
            const uchar* srcPtr = channels[1].ptr<uchar>(row);
            uchar* dstPtr = dst.ptr<uchar>(row);
            for (int col = 0; col < channels[1].cols; ++col) {
                dstPtr[col] = static_cast<uchar>(cdfNormalized[srcPtr[col]]);
            }
        }
        channels[1] = dst;

        // HSL�� �ٽ� BGR�� ��ȯ
        hsl = merge_Serial(channels[2], channels[1], channels[0]);
        cvtColor(hsl, dst, COLOR_HLS2BGR);
    }

    //HSL����ó�� �Լ�
    void equalizeHistUsingHSL_Parallel(Mat& src, Mat& dst) {
        vector<Mat> channels;
        Mat hsl;
        vector<int> cdf(256, 0);

        // BGR���� HSL�� ��ȯ
        cvtColor(src, hsl, COLOR_BGR2HLS);

        // HSLä�� �迭�� ���� ������
        split_Parallel(hsl, channels);

        // L ������׷� Ȯ��
        vector<int> hist;
        CustomcalcHist_Parallel(channels[1], hist);
        dst = Mat(channels[1].rows, channels[1].cols, CV_8UC1);

        // ������׷� ����ȭ
        cdf[0] = hist[0];
        int totalPixels = channels[1].rows * channels[1].cols;
        vector<int> cdfNormalized(256, 0);
        cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);

        /* ����ó���� �ʿ��� ���� ������ ���� ����� ���� ����� ������ �־� ����ȭ �Ұ���*/
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
            cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
        }

        // L������׷� ��Ȱȭ   
#pragma omp parallel for num_threads(can_div_threads)
        for (int row = 0; row < channels[1].rows; ++row) {
            const uchar* srcPtr = channels[1].ptr<uchar>(row);
            uchar* dstPtr = dst.ptr<uchar>(row);
            for (int col = 0; col < channels[1].cols; ++col) {
                dstPtr[col] = static_cast<uchar>(cdfNormalized[srcPtr[col]]);
            }
        }
        channels[1] = dst;

        // HSL�� �ٽ� BGR�� ��ȯ
        hsl = merge_Parallel(channels[2], channels[1], channels[0]);
        cvtColor(hsl, dst, COLOR_HLS2BGR);
    }

    //���߿� ���� �̹��� �ΰ��� ������ ��ġ�� �Լ�
    void CustomaddWeighted_Serial(const Mat& d1, double alpha, const Mat& d2, double beta, Mat& dst) {

        dst.create(d1.size(), d1.type());


        for (int y = 0; y < d1.rows; ++y) {
            const uchar* row_d1 = d1.ptr(y);
            const uchar* row_d2 = d2.ptr(y);
            uchar* row_dst = dst.ptr(y);

            for (int x = 0; x < d1.cols; ++x) {
                row_dst[3 * x + 0] = saturate_cast<uchar>(alpha * row_d1[3 * x + 0] + beta * row_d2[3 * x + 0]);
                row_dst[3 * x + 1] = saturate_cast<uchar>(alpha * row_d1[3 * x + 1] + beta * row_d2[3 * x + 1]);
                row_dst[3 * x + 2] = saturate_cast<uchar>(alpha * row_d1[3 * x + 2] + beta * row_d2[3 * x + 2]);
            }
        }
    }

    void CustomaddWeighted_Parallel(const Mat& d1, double alpha, const Mat& d2, double beta, Mat& dst) {

        dst.create(d1.size(), d1.type());

#pragma omp parallel for
        for (int y = 0; y < d1.rows; ++y) {
            const uchar* row_d1 = d1.ptr(y);
            const uchar* row_d2 = d2.ptr(y);
            uchar* row_dst = dst.ptr(y);

            for (int x = 0; x < d1.cols; ++x) {
                row_dst[3 * x + 0] = saturate_cast<uchar>(alpha * row_d1[3 * x + 0] + beta * row_d2[3 * x + 0]);
                row_dst[3 * x + 1] = saturate_cast<uchar>(alpha * row_d1[3 * x + 1] + beta * row_d2[3 * x + 1]);
                row_dst[3 * x + 2] = saturate_cast<uchar>(alpha * row_d1[3 * x + 2] + beta * row_d2[3 * x + 2]);
            }
        }
    }
}