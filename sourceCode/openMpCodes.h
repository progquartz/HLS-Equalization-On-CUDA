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

        //OpenCV BGR값을 B, G, R로 분할하여 저장
        split_Serial(input_image, bgr_planes);

        //BGR값에 대하여 히스토그램 평탄화 진행
        for (int i = 0; i < 3; i++)
        {
            equalizeHistBGR_Serial(bgr_planes[i], bgr_planes_results[i]);
        }

        //BGR값을 다시 하나의 이미지로 합치기
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

        //병렬 중첩 허용
        omp_set_nested(1);

        //section으로 HSL과 BGR의 처리를 병렬적으로 실행하도록 함.
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

    //BGR값에 대한 히스토그램 평활화 함수 병렬
    void equalizeHistUsingBGR_Parallel(Mat& src, Mat& dst) {
        vector<Mat> bgr_planes;
        vector<Mat> bgr_planes_results(3);

        split_Parallel(src, bgr_planes);

        //RGB 세개의 배열에 대해서 진행하므로 Thread의 개수 3
#pragma omp parallel for num_threads(3)
        for (int i = 0; i < 3; i++)
        {
            //BGR값에 대하여 히스토그램 평탄화 진행
            equalizeHistBGR_Parallel(bgr_planes[i], bgr_planes_results[i]);
        }
        dst = merge_Parallel(bgr_planes_results[2], bgr_planes_results[1], bgr_planes_results[0]);
    }


    //BGR히스토그램 기본 평렬화 함수
    void equalizeHistBGR_Serial(Mat& src, Mat& dst) {
        //히스토그램 계산
        vector<int> hist;
        CustomcalcHist_Serial(src, hist, omp_get_thread_num());

        //히스토그램 정규화
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
        //히스토그램 평활화
        for (int row = 0; row < src.rows; ++row) {
            const uchar* srcPtr = src.ptr<uchar>(row);
            uchar* dstPtr = dst.ptr<uchar>(row); // dstPtr 타입 변경
            for (int col = 0; col < src.cols; ++col) {
                dstPtr[col] = static_cast<uchar>(cdfNormalized[srcPtr[col]]); // 픽셀 값을 평활화된 값으로 변경
            }
        }
    }

    //BGR히스토그램 병렬 평렬화 함수
    void equalizeHistBGR_Parallel(Mat& src, Mat& dst) {
        // 히스토그램 계산
        vector<int> hist;
        CustomcalcHist_Parallel(src, hist, omp_get_thread_num());

        vector<int> cdf(256, 0);
        cdf[0] = hist[0];
        int totalPixels = src.rows * src.cols;
        vector<int> cdfNormalized(256, 0);

        //히스토그램 정규화
        cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
            cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
        }

        dst = Mat(src.rows, src.cols, src.type());
        //히스토그램 평활화
#pragma omp parallel for num_threads(can_div_threads)
        for (int row = 0; row < src.rows; ++row) {
            const uchar* srcPtr = src.ptr<uchar>(row);
            uchar* dstPtr = dst.ptr<uchar>(row); // dstPtr 타입 변경
            for (int col = 0; col < src.cols; ++col) {
                dstPtr[col] = static_cast<uchar>(cdfNormalized[srcPtr[col]]); // 픽셀 값을 평활화된 값으로 변경
            }
        }
    }

    //분해 기본 함수
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

    //분해 병렬 함수
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


    //merge 시리얼 함수
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

    //merge 병렬함수
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

    //히스토그램 파악 기본 함수
    void CustomcalcHist_Serial(Mat& src, vector<int>& hist, int thread = -1) {
        hist = vector<int>(256, 0);
        for (int row = 0; row < src.rows; ++row) {
            unsigned char* ptr = src.ptr<uchar>(row);
            for (int col = 0; col < src.cols; ++col) {
                hist[ptr[col]]++;
            }
        }
    }

    //히스토그램 파악 병렬 함수
    void CustomcalcHist_Parallel(Mat& src, vector<int>& hist, int thread = -1) {
        hist = vector<int>(256, 0);
#pragma omp parallel num_threads(can_div_threads*2)
        {
            /* no wait을 사용하여 자신의 구간이 끝난 것은 바로 결과값에 포함되도록 함*/
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


    //HSL기본처리 함수
    void equalizeHistUsingHSL_Serial(Mat& src, Mat& dst) {
        Mat hsl;
        vector<Mat> channels;
        vector<int> hist;

        // BGR에서 HSL로 변환
        cvtColor(src, hsl, COLOR_BGR2HLS);

        // HSL채널 배열에 각각 나누기
        split_Serial(hsl, channels);

        // L 히스토그램 확인
        CustomcalcHist_Serial(channels[1], hist);

        // 히스토그램 정규화
        vector<int> cdf(256, 0);
        cdf[0] = hist[0];
        int totalPixels = channels[1].rows * channels[1].cols;
        vector<int> cdfNormalized(256, 0);
        cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
            cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
        }

        // L히스토그램 평활화
        dst = Mat(channels[1].rows, channels[1].cols, CV_8UC1);
        for (int row = 0; row < channels[1].rows; ++row) {
            const uchar* srcPtr = channels[1].ptr<uchar>(row);
            uchar* dstPtr = dst.ptr<uchar>(row);
            for (int col = 0; col < channels[1].cols; ++col) {
                dstPtr[col] = static_cast<uchar>(cdfNormalized[srcPtr[col]]);
            }
        }
        channels[1] = dst;

        // HSL을 다시 BGR로 변환
        hsl = merge_Serial(channels[2], channels[1], channels[0]);
        cvtColor(hsl, dst, COLOR_HLS2BGR);
    }

    //HSL병렬처리 함수
    void equalizeHistUsingHSL_Parallel(Mat& src, Mat& dst) {
        vector<Mat> channels;
        Mat hsl;
        vector<int> cdf(256, 0);

        // BGR에서 HSL로 변환
        cvtColor(src, hsl, COLOR_BGR2HLS);

        // HSL채널 배열에 각각 나누기
        split_Parallel(hsl, channels);

        // L 히스토그램 확인
        vector<int> hist;
        CustomcalcHist_Parallel(channels[1], hist);
        dst = Mat(channels[1].rows, channels[1].cols, CV_8UC1);

        // 히스토그램 정규화
        cdf[0] = hist[0];
        int totalPixels = channels[1].rows * channels[1].cols;
        vector<int> cdfNormalized(256, 0);
        cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);

        /* 병렬처리가 필요할 수도 있으나 이전 결과가 다음 결과에 영향을 주어 병렬화 불가능*/
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hist[i];
            cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
        }

        // L히스토그램 평활화   
#pragma omp parallel for num_threads(can_div_threads)
        for (int row = 0; row < channels[1].rows; ++row) {
            const uchar* srcPtr = channels[1].ptr<uchar>(row);
            uchar* dstPtr = dst.ptr<uchar>(row);
            for (int col = 0; col < channels[1].cols; ++col) {
                dstPtr[col] = static_cast<uchar>(cdfNormalized[srcPtr[col]]);
            }
        }
        channels[1] = dst;

        // HSL을 다시 BGR로 변환
        hsl = merge_Parallel(channels[2], channels[1], channels[0]);
        cvtColor(hsl, dst, COLOR_HLS2BGR);
    }

    //비중에 따라 이미지 두개의 정보를 합치는 함수
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