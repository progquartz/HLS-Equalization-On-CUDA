#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "DS_timer.h"
#include "DS_definitions.h"
#include <math.h>
#include "openMpCodes.h"

#include <vector>
#include <cmath>

using namespace cv;

int width;
int height;

/// <summary>
/// *********************************************************************************
/// 
/// 여기에서부터 다음 이 포멧이 나오기 전까지는, HLS에 대한 함수들입니다.
/// 
/// HLS에 대한 평활화는 3가지 과정으로 진행됩니다.
/// 이 과정은 HLS 파트의 가장 아래 함수인 HLS_histogramEqualizationCUDA에서 진행됩니다.
/// 
/// 1. HLS데이터 포멧을 받아오기.
/// 2. 받아온 L데이터에 대한 히스토그램 평활화를 진행
/// 3. 그 결과를 저장.
/// 
/// 중요한 함수의 경우, //로 처리되며, 중요한 함수의 경우 ///로 확인하실 수 있을겁니다.
/// *********************************************************************************
/// </summary>



// HLS에 대한 히스토그램의 계산을 합니다. 히스토그램의 'L의 수' 번째 인덱스를 1 더합니다.
__global__ void HLS_histogramEqualization(const uchar* input, uchar* output, int* hist, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 각 픽셀에 대해
    if (x < width && y < height)
    {
        int index = (y * width + x);
        // Compute histogram
        // input의 L의 값을 받아와 이에 대해 
        atomicAdd(&hist[input[index]], 1);
    }
}

/// HLS의 L에 대한 히스토그램 평활화를 합니다. 
void HLS_histogramNormalize(int* hist, int* cdfNormalized, int totalPixels) {
    //히스토그램 정규화
    std::vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);

    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
        cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
    }
}

/// HLS에 대한 L의 평활화된 데이터를 기반으로 output의 값을 가져옵니다.
__global__ void HLS_Equalization(const uchar* input, uchar* output, int* norhist, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x);
        // output에 평활화된 값을 넣기 (uchar)로 static_cast 함.
        output[index] = static_cast<uchar>(norhist[input[index]]);
    }
}

/// 다음 함수는, 받아온 input_image에 대한 히스토그램 평활화를 진행하고, 그 결과를 output_image에 넣습니다.
/// 1. RGB데이터 포멧을 HLS데이터 포멧으로 바꾸기.
/// 2. HLS 데이터를 Split 하여 L(Lightness)만을 가져와 이에 대한 평활화를 진행합니다.
/// 3. HLS데이터를 다시 합친 뒤, 이를 RGB 포멧으로 변환하여 결과값을 만들어냅니다.
void HLS_histogramEqualizationCUDA(const cv::Mat& input_image, cv::Mat& output_image)
{
    int width = input_image.cols;
    int height = input_image.rows;
    int channels = input_image.channels();

    // 히스토그램 공간
    int hist[256];
    int norHist[256];

    // GPU 히스토그램 공간
    int* d_hist;
    int* d_norHist;

    cudaMalloc(&d_hist, sizeof(int) * 256);
    cudaMalloc(&d_norHist, sizeof(float) * 256);
    cudaMemset(&d_hist, 0, sizeof(int) * 256);
    cudaMemset(&d_norHist, 0.0f, sizeof(float) * 256);

    // GPU 메모리 선언
    uchar* d_input_image;
    uchar* d_output_image;

    cudaMalloc((void**)&d_input_image, input_image.total());
    cudaMalloc((void**)&d_output_image, input_image.total());
    cudaMemcpy(d_input_image, input_image.data, input_image.total(), cudaMemcpyHostToDevice);

    // 그리드와 블럭 생성.
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // HLS에 대한 히스토그램의 계산을 합니다. 히스토그램의 'L의 수' 번째 인덱스를 1 더합니다.
    HLS_histogramEqualization << <gridSize, blockSize >> > (d_input_image, d_output_image, d_hist, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(hist, d_hist, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    /// HLS의 L에 대한 히스토그램 평활화를 합니다. 이는 norhist에 저장됩니다.
    HLS_histogramNormalize(&hist[0], norHist, width * height);

    // norhist에 저장되었던 값을 d_norHist로 올려 데이터를 계산할 수 있도록 합니다.
    cudaMemcpy(d_norHist, norHist, sizeof(int) * 256, cudaMemcpyHostToDevice);

    /// HLS에 대한 L의 평활화된 데이터를 기반으로 output의 값을 가져옵니다.
    HLS_Equalization << < gridSize, blockSize >> > (d_input_image, d_output_image, d_norHist, width, height, channels);
    cudaDeviceSynchronize();

    // 결과를 outputimage에 둡니다.
    cudaMemcpy(output_image.data, d_output_image, output_image.total(), cudaMemcpyDeviceToHost);

    // 메모리 free
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    cudaFree(d_norHist);
    cudaFree(d_hist);
}

/// <summary>
/// *********************************************************************************
/// 
/// 여기에서부터 다음 이 포멧이 나오기 전까지는, RGB에 대한 함수들입니다.
/// 
/// RGB에 대한 평활화는 3가지 과정으로 진행됩니다.
/// 이 과정은 RGB 파트의 가장 아래 함수인 RGB_histogramEqualizationCUDA에서 진행됩니다.
/// 
/// 1. RGB데이터 포멧을 받아오기.
/// 2. 받아온 RGB 각각에 대한 히스토그램 평활화를 진행
/// 3. 그 결과를 저장.
/// 
/// 중요한 함수의 경우, //로 처리되며, 중요한 함수의 경우 ///로 확인하실 수 있을겁니다.
/// *********************************************************************************
/// </summary>

// RGB에 대한 히스토그램의 계산을 합니다. 히스토그램의 'RGB 각각의 수' 번째 인덱스를 1 더합니다.
__global__ void RGB_histogramEqualization(const uchar* input, uchar* output, int* hist, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * channels;

        // 히스토그램의 인덱스 1씩 더하기.
        atomicAdd(&hist[input[index]], 1);
        atomicAdd(&hist[input[index + 1] + 256], 1);
        atomicAdd(&hist[input[index + 2] + 512], 1);
    }
}

/// RGB의 RGB에 대한 히스토그램 평활화를 합니다. 
void RGB_histogramNormalize(int* hist, int* cdfNormalized, int totalPixels) {
    //히스토그램 평활화
    std::vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255.0f);

    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
        cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255.0f);
    }
}

/// RGB에 대한 평활화된 데이터(norhist)를 기반으로 output의 값을 가져옵니다.
__global__ void RGB_Equalization(const uchar* input, uchar* output, int* norhist, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 각 좌표에 대해
    if (x < width && y < height)
    {
        int index = (y * width + x) * channels;

        output[index] = static_cast<uchar>(norhist[input[index]]);
        output[index + 1] = static_cast<uchar>(norhist[input[index + 1] + 256]);
        output[index + 2] = static_cast<uchar>(norhist[input[index + 2] + 512]);
    }
}


/// 다음 함수는, 받아온 input_image에 대한 히스토그램 평활화를 진행하고, 그 결과를 output_image에 넣습니다.
/// 1. RGB데이터 포멧을 받아오기.
/// 2. 받아온 RGB 각각에 대한 히스토그램 평활화를 진행
/// 3. 그 결과를 저장.
void RGB_histogramEqualizationCUDA(const cv::Mat& input_image, cv::Mat& output_image)
{
    int width = input_image.cols;
    int height = input_image.rows;
    int channels = input_image.channels();

    // 히스토그램 공간
    int hist[768];
    int norHist[768];

    // GPU 리스토그램 공간
    int* d_hist;
    int* d_norHist;

    cudaMalloc(&d_hist, sizeof(int) * 768);
    cudaMemset(&d_hist, 0, sizeof(int) * 768);
    cudaMalloc(&d_norHist, sizeof(int) * 768);

    // GPU 메모리 선언.
    uchar* d_input_image;
    uchar* d_output_image;

    cudaMalloc((void**)&d_input_image, input_image.total() * channels);
    cudaMalloc((void**)&d_output_image, input_image.total() * channels);
    cudaMemcpy(d_input_image, input_image.data, input_image.total() * channels, cudaMemcpyHostToDevice);

    // 블록과 그리드 생성.
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // RGB에 대한 히스토그램의 계산을 합니다. 히스토그램의 'RGB 각각의 수' 번째 인덱스를 1 더합니다.
    RGB_histogramEqualization << <gridSize, blockSize >> > (d_input_image, d_output_image, d_hist, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(hist, d_hist, sizeof(int) * 768, cudaMemcpyDeviceToHost);

    /// RGB에 대한 히스토그램 평활화를 합니다. 이는 norhist에 저장됩니다.
    RGB_histogramNormalize(hist, norHist, width * height);
    RGB_histogramNormalize(&hist[256], &norHist[256], width * height);
    RGB_histogramNormalize(&hist[512], &norHist[512], width * height);

    // norhist에 저장되었던 값을 d_norHist로 올려 데이터를 계산할 수 있도록 합니다.
    cudaMemcpy(d_norHist, norHist, sizeof(int) * 768, cudaMemcpyHostToDevice);

    /// RGB의 평활화된 데이터를 기반으로 output의 값을 가져옵니다.
    RGB_Equalization << < gridSize, blockSize >> > (d_input_image, d_output_image, d_norHist, width, height, channels);
    cudaDeviceSynchronize();

    // 결과를 outputimage에 둡니다.
    cudaMemcpy(output_image.data, d_output_image, output_image.total() * channels, cudaMemcpyDeviceToHost);

    // 메모리 Free
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    cudaFree(d_norHist);
    cudaFree(d_hist);
}


/// <summary>
/// *********************************************************************************
/// 
/// 여기에서부터 다음 이 포멧이 나오기 전까지는, 모든 부분을 합치는 부분입니다.
/// 
/// 이는 각각 HLS와 RGB가 개별적으로 진행되며, RGB의 경우 추가 연산이 없지만 
/// HLS의 경우 RGB -> HLS / HLS -> RGB 연산을 추가로 요합니다.
/// 
/// 마지막으로 결과값은 RGB와 HLS를 반반씩 섞어 진행하게 됩니다. (비율 조절해서 더 나은 결과 가능할 것 같음)
/// 
/// 중요한 함수의 경우, //로 처리되며, 중요한 함수의 경우 ///로 확인하실 수 있을겁니다.
/// *********************************************************************************
/// </summary>


// float값들에 대한 mod값을 구하는 함수.
__device__ float customFmodf(float dividend, float divisor) {
    return dividend - divisor * floorf(dividend / divisor);
}

__device__ float HueToRGB(float v1, float v2, float vH) {
    if (vH < 0)
        vH += 1;

    if (vH > 1)
        vH -= 1;

    if ((6 * vH) < 1)
        return (v1 + (v2 - v1) * 6 * vH);

    if ((2 * vH) < 1)
        return v2;

    if ((3 * vH) < 2)
        return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

    return v1;
}
/// 주어진 uchar* 형식으로 주어진 HSL 데이터(input)을 RGB로 변환해 output에 저장하는 함수.
// 참조 1 https://www.rapidtables.com/convert/color/hsl-to-rgb.html
// 참조 2 http://dystopiancode.blogspot.com/2012/06/hsl-rgb-conversion-algorithms-in-c.html (많이 복잡함)
// 참조 3 https://codegolf.stackexchange.com/questions/150250/hsl-to-rgb-values (색공간에 대한 이해가 있으면 가장 쉬운 이해가 가능)
__global__ void ConvertHLSToRGB(const uchar* input, uchar* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size = width * height;

    // 각 픽셀에 대해 연산이 진행됩니다.
    if (x < width && y < height) {

        int index = (y * width + x) * channels;

        float r, g, b;

        // 기존에 0~1값을 가지던 hue를 실제 0도에서 360사이의 값이 가지도록 설정.
        float hue = input[index] / 255*360 ;
        float lightness = input[index + 1] /255;
        float saturation = input[index + 2] / 255;

        if (saturation == 0)
        {
            r = (unsigned char)(lightness * 255);
            g = (unsigned char)(lightness * 255);
            b = (unsigned char)(lightness * 255);
        }
        else
        {
            float v1, v2;

            if (lightness < 0.5) {
                v2 = lightness * (1 + saturation);
            }
            else {
                v2 = lightness + saturation - (lightness * saturation);
            }
            v1 = 2 * lightness - v2;
            
            if (saturation == 0)
            {
                r = g = b = (unsigned char)(lightness * 255);
            }
            else
            {
                float v1, v2;
                float hue = (float)hue / 360;

                v2 = (lightness < 0.5) ? (lightness * (1 + saturation)) : ((lightness + saturation) - (lightness * saturation));
                v1 = 2 * lightness - v2;

                r = (unsigned char)(255 * HueToRGB(v1, v2, hue + (1.0f / 3)));
                g = (unsigned char)(255 * HueToRGB(v1, v2, hue));
                b = (unsigned char)(255 * HueToRGB(v1, v2, hue - (1.0f / 3)));
            }

            output[index] = static_cast<uchar>(b);
            output[index + 1] = static_cast<uchar>(g);
            output[index + 2] = static_cast<uchar>(r);

        }

    }
}

__device__ float Min(float a, float b) {
    return a <= b ? a : b;
}

__device__ float Max(float a, float b) {
    return a >= b ? a : b;
}

/// 주어진 uchar* 형식으로 주어진 RGB 데이터(input)을 HLS로 변환해 output에 저장하는 함수. 
// 참조 1 : https://stackoverflow.com/questions/39118528/rgb-to-hsl-conversion
__global__ void ConvertRGBToHLS(const uchar* input, uchar* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size = width * height;

    float lightness;
    float saturation;
    float hue;

    // 각 픽셀에 대해 연산.
    if (x < width && y < height) {

        int index = (y * width + x) * channels;

        double b = input[index] / 255.0f;
        double g = input[index + 1] / 255.0f;
        double r = input[index + 2] / 255.0f;

        float min = Min(Min(r, g), b);
        float max = Max(Max(r, g), b);
        float delta = max - min;

        lightness = (max + min) / 2;

        if (delta == 0)
        {
            hue = 0;
            saturation = 0.0f;
        }
        else
        {
            saturation = (lightness <= 0.5) ? (delta / (max + min)) : (delta / (2 - max - min));

            float hue;

            if (r == max)
            {
                hue = ((g - b) / 6) / delta;
            }
            else if (g == max)
            {
                hue = (1.0f / 3) + ((b - r) / 6) / delta;
            }
            else
            {
                hue = (2.0f / 3) + ((r - g) / 6) / delta;
            }

            if (hue < 0)
                hue += 1;
            if (hue > 1)
                hue -= 1;
        }

        // 저장된 값을 저장.
        output[index] = static_cast<uchar>(hue * 255);
        output[index + 1] = static_cast<uchar>(lightness * 255);
        output[index + 2] = static_cast<uchar>(saturation * 255);
    }
}


/// 다음 함수는 RGB를 기반으로 input1에 대한 alpha의 가중치를, input2에 대한 beta의 가중치를 가져 섞은 결과를 output에 저장하는 커널입니다.
__global__ void CustomWeight(const uchar* input1, double alpha, const uchar* input2, double beta, uchar* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * channels;

        // 각각 RGB의 데이터를 (BGR) 비율에 맞게 섞어서 static_cast 시킵니다.
        output[index] = static_cast<uchar>((alpha * input1[index]) + (beta * input2[index]));
        output[index + 1] = static_cast<uchar>((alpha * input1[index + 1]) + (beta * input2[index + 1]));
        output[index + 2] = static_cast<uchar>((alpha * input1[index + 2]) + (beta * input2[index + 2]));
    }
}

/// 다음은 주어진 2개의 이미지를 받아 이를 alpha와 beta의 비로 섞어 결과를 내는 함수입니다.
void CustomAddWeighted_CUDA(const Mat& input_image1, double alpha, const Mat& input_image2, double beta, Mat& output_image)
{
    int width = input_image1.cols;
    int height = input_image1.rows;
    int channels = input_image1.channels();
    output_image.create(input_image1.size(), input_image1.type());

    uchar* d_input_image1;
    uchar* d_input_image2;
    uchar* d_output_image;

    cudaMalloc((void**)&d_input_image1, input_image1.total() * channels);
    cudaMalloc((void**)&d_input_image2, input_image2.total() * channels);
    cudaMalloc((void**)&d_output_image, input_image1.total() * channels);

    cudaMemcpy(d_input_image1, input_image1.data, input_image1.total() * channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_image2, input_image2.data, input_image2.total() * channels, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    /// 합치는 연산
    CustomWeight << <gridSize, blockSize >> > (d_input_image1, alpha, d_input_image2, beta, d_output_image, width, height, channels);

    cudaDeviceSynchronize();
    cudaMemcpy(output_image.data, d_output_image, output_image.total() * channels, cudaMemcpyDeviceToHost);

    cudaFree(d_input_image1);
    cudaFree(d_input_image2);
    cudaFree(d_output_image);

}

/// 다음 코드는 한 프레임의 영상에 대해 RGB값이 평활화되는 함수입니다.
void RGBGetWeightedImage(const Mat& input_image, Mat& output_image)
{
    RGB_histogramEqualizationCUDA(input_image, output_image);
}

/// 다음 코드는 한 프레임의 영상에 대해 RGB와 HSL값이 전부 평활화되는 함수입니다.
void RGBHSLGetWeightedImage(const Mat& input_image, Mat& output_image)
{
    /// RGB part
    cv::Mat RGB_output_image(input_image.size(), input_image.type());
    RGB_histogramEqualizationCUDA(input_image, RGB_output_image);


    /// HSL part
    cv::Mat hslImage(input_image.size(), input_image.type());
    cv::Mat out(input_image.size(), input_image.type());
    int imageType = input_image.type();

    /*cvtColor 추가한 부분*/
    int width = input_image.cols;
    int height = input_image.rows;
    int channel = input_image.channels();

    uchar* d_input_image;
    uchar* d_output_image;
    cudaMalloc((void**)&d_input_image, input_image.total() * channel);
    cudaMalloc((void**)&d_output_image, input_image.total() * channel);

    cudaMemcpy(d_input_image, input_image.data, input_image.total() * channel, cudaMemcpyHostToDevice);

    // 블록과 그리드 선언.
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    /// RGB데이터를 HLS로 변환
    ConvertRGBToHLS << <gridSize, blockSize >> > (d_input_image, d_output_image, width, height, channel);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data, d_output_image, input_image.total() * channel, cudaMemcpyDeviceToHost);

    hslImage.data = out.data;

    //cv::cvtColor(input_image, hslImage, cv::COLOR_BGR2HLS);
    // 변환된 데이터를 hslImage에 저장.
    // hsl로 변환된 데이터를 쪼개서 h, s, l 각각에 대한 Mat로 나눔.

    std::vector<cv::Mat> channels(3);
    cv::split(hslImage, channels);

    /// 변환된 HLS에 대한 L값에 대한 평활화 진행.
    HLS_histogramEqualizationCUDA(channels[1], channels[1]);


    // L값이 평활화되었으므로, 다시 HLS로 Merge.
    cv::merge(channels, hslImage);

    cv::Mat HSL_output_image(input_image.size(), input_image.type());
    cudaMemcpy(d_input_image, hslImage.data, input_image.total() * channel, cudaMemcpyHostToDevice);

    /// 평활화된 HLS 데이터를 다시 RGB로 변환
    ConvertHLSToRGB << <gridSize, blockSize >> > (d_input_image, d_output_image, width, height, channel);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data, d_output_image, input_image.total() * channel, cudaMemcpyDeviceToHost);


    // 평활화된 데이터가 RGB로 변환된것을 저장.
    HSL_output_image.data = out.data;

    //cv::cvtColor(hslImage, HSL_output_image, cv::COLOR_HLS2BGR);

    /// 두 결과 (RGB 평활화 결과 / HLS 평활화 결과) 색깔 합치기
    CustomAddWeighted_CUDA(HSL_output_image, 0.5, RGB_output_image, 0.5, output_image);
}

/// 다음 코드는 한 프레임의 영상에 대해 RGB와 HSL값이 전부 평활화되는 함수입니다.
void RGBHSLGetWeightedImage_CV(const Mat & input_image, Mat & output_image)
{
    /// RGB part
    cv::Mat RGB_output_image(input_image.size(), input_image.type());
    RGB_histogramEqualizationCUDA(input_image, RGB_output_image);

    /// HSL part
    cv::Mat hslImage;
    int imageType = input_image.type();

    cv::cvtColor(input_image, hslImage, cv::COLOR_BGR2HLS);
    // Split channels
    std::vector<cv::Mat> channels(3);
    cv::split(hslImage, channels);
    HLS_histogramEqualizationCUDA(channels[1], channels[1]);
    cv::merge(channels, hslImage);


    cv::Mat HSL_output_image;
    cv::cvtColor(hslImage, HSL_output_image, cv::COLOR_HLS2BGR);

    /// 색깔 합치기
    CustomAddWeighted_CUDA(HSL_output_image, 0.5, RGB_output_image, 0.5, output_image);
}


int main(int argc, char* argv[]) {
    // 동영상 파일 열기
    DS_timer timer(6);
    timer.setTimerName(1, "[RGB OnlyCounter]");
    timer.setTimerName(2, "[SerialCounter]");
    timer.setTimerName(3, "[ParallelCounter]");
    timer.setTimerName(4, "[Slow Cuda Counter]");
    timer.setTimerName(5, "[both Cuda Counter]");
    timer.initTimers();

    string filename = argv[1];
    cv::VideoCapture capRGBCUDA(filename);
    cv::VideoCapture capBothCUDA(filename);
    cv::VideoCapture capSlowCUDA(filename);
    cv::VideoCapture capParallel(filename);
    cv::VideoCapture capSerial(filename);
    if (!capRGBCUDA.isOpened()) {
        std::cout << "동영상 파일을 열 수 없습니다." << std::endl;
        return -1;
    }

    // 동영상의 속성 가져오기
    int frame_width = static_cast<int>(capRGBCUDA.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(capRGBCUDA.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capRGBCUDA.get(cv::CAP_PROP_FPS);
    int num_frames = static_cast<int>(capRGBCUDA.get(cv::CAP_PROP_FRAME_COUNT));

    // 비디오 라이터 생성
    cv::VideoWriter outputSerial("outputSerial.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    cv::VideoWriter outputParallel("outputParallel.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    cv::VideoWriter outputRGBCUDA("outputRGBCUDA.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    cv::VideoWriter outputBothCUDA("outputHSLRGBCUDA.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    cv::VideoWriter outputSlowCUDA("outputSlowCUDA.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));

    int i = 0;

    /// Serial (그냥 CPU 버전)
    cv::Mat frameSerial, frame_hist_equalized_Serial;
    i = 0;
    while (true) {
        capSerial >> frameSerial;
        if (frameSerial.empty())
            break;

        frame_hist_equalized_Serial.copySize(frameSerial);

        timer.onTimer(2);
        // 각 채널에 대해 히스토그램 평활화 수행
        pre::histogramEqualizationSerial(frameSerial, frame_hist_equalized_Serial);
        timer.offTimer(2);
        std::cout << "Serial - " << i++ << "번째 프레임" << "\n";

        // 결과 프레임을 비디오 라이터에 저장
        outputSerial.write(frame_hist_equalized_Serial);
    }


    /// Parallel (Open MP 버전)

    cv::Mat frameParallel, frame_hist_equalized_Parallel;
    i = 0;
    while (true) {
        capParallel >> frameParallel;
        if (frameParallel.empty())
            break;

        frame_hist_equalized_Parallel.copySize(frameParallel);

        timer.onTimer(3);
        // 각 채널에 대해 히스토그램 평활화 수행
        pre::HistogramEqualizationOpenMP(frameParallel, frame_hist_equalized_Parallel);
        timer.offTimer(3);
        std::cout << "Parallel - " << i++ << "번째 프레임" << "\n";

        // 결과 프레임을 비디오 라이터에 저장
        outputParallel.write(frame_hist_equalized_Parallel);
    }
    

    /// 느린 CUDA 버전
    i = 0;
    cv::Mat frameSlowCUDA, frame_hist_equalized_Slow_CUDA;
    while (true) {
        capSlowCUDA >> frameSlowCUDA;
        if (frameSlowCUDA.empty())
            break;
        // 디바이스에서 호스트로 프레임 복사
        std::vector<cv::Mat> channels;
        cv::split(frameSlowCUDA, channels);

        // 각 채널에 대해 히스토그램 평활화 수행
        timer.onTimer(4);
        RGBHSLGetWeightedImage_CV(frameSlowCUDA, frame_hist_equalized_Slow_CUDA);
        timer.offTimer(4);
        std::cout << "CV Converting CUDA - " << i++ << "번째 프레임" << "\n";

        // 결과 프레임을 비디오 라이터에 저장
        outputSlowCUDA.write(frame_hist_equalized_Slow_CUDA);
    }

    /// RGB CUDA 버전
    i = 0;
    cv::Mat frameRGBCUDA, frame_hist_equalized_RGB_CUDA;
    while (true) {
        capRGBCUDA >> frameRGBCUDA;
        if (frameRGBCUDA.empty())
            break;
        // 디바이스에서 호스트로 프레임 복사
        std::vector<cv::Mat> channels;
        cv::split(frameRGBCUDA, channels);
        frame_hist_equalized_RGB_CUDA = Mat(frameRGBCUDA.size(), frameRGBCUDA.type());
        // 각 채널에 대해 히스토그램 평활화 수행
        timer.onTimer(1);
        RGBGetWeightedImage(frameRGBCUDA, frame_hist_equalized_RGB_CUDA);
        timer.offTimer(1);
        std::cout << "RGB CUDA - " << i++ << "번째 프레임" << "\n";

        // 결과 프레임을 비디오 라이터에 저장
        outputRGBCUDA.write(frame_hist_equalized_RGB_CUDA);
    }

    /// HLS / RGB 이용한 버전
    i = 0;
    cv::Mat frameBothCUDA, frame_hist_equalized_Both_CUDA;
    while (true) {
        capBothCUDA >> frameBothCUDA;
        if (frameBothCUDA.empty())
            break;
        // 디바이스에서 호스트로 프레임 복사
        std::vector<cv::Mat> channels;
        cv::split(frameBothCUDA, channels);

        // 각 채널에 대해 히스토그램 평활화 수행
        timer.onTimer(5);
        RGBHSLGetWeightedImage(frameBothCUDA, frame_hist_equalized_Both_CUDA);
        timer.offTimer(5);
        std::cout << "HLS / RGB CUDA - " << i++ << "번째 프레임" << "\n";

        // 결과 프레임을 비디오 라이터에 저장
        outputBothCUDA.write(frame_hist_equalized_Both_CUDA);
    }


    // 비디오 파일과 비디오 라이터 닫기
    capRGBCUDA.release();
    capBothCUDA.release();
    capSlowCUDA.release();
    capParallel.release();
    capSerial.release();

    outputSerial.release();
    outputParallel.release();
    outputSlowCUDA.release();
    outputBothCUDA.release();
    outputRGBCUDA.release();


    std::cout << "히스토그램 평활화된 동영상이 output.mp4 파일로 저장되었습니다." << std::endl;
    std::cout << "Serial 버전 전체 프레임" << i << "수행 시간 : " << timer.getTimer_ms(2) << "ms" << std::endl;
    std::cout << "Serial 버전 한 프레임당 계산 시간 : " << timer.getTimer_ms(2) / i << "ms" << std::endl;
    std::cout << "Parallel 버전 전체 프레임" << i << "수행 시간 : " << timer.getTimer_ms(3) << "ms" << std::endl;
    std::cout << "Parallel 버전 한 프레임당 계산 시간 : " << timer.getTimer_ms(3) / i << "ms" << std::endl;
    std::cout << "CUDA CV Converting 버전 전체 프레임 : " << i << " 수행 시간 : " << timer.getTimer_ms(4) << "ms" << std::endl;
    std::cout << "CUDA CV Converting 버전 한 프레임당 계산 시간 : " << timer.getTimer_ms(4) / i << "ms" << std::endl;
    std::cout << "CUDA RGB 버전 전체 프레임 : " << i << " 수행 시간 : " << timer.getTimer_ms(1) << "ms" << std::endl;
    std::cout << "CUDA RGB 버전 한 프레임당 계산 시간 : " << timer.getTimer_ms(1) / i << "ms" << std::endl;
    std::cout << "CUDA HSL / RGB 버전 전체 프레임 : " << i << " 수행 시간 : " << timer.getTimer_ms(5) << "ms" << std::endl;
    std::cout << "CUDA HSL / RGB 버전 한 프레임당 계산 시간 : " << timer.getTimer_ms(5) / i << "ms" << std::endl;
    return 0;
}