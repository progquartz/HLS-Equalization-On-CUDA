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
/// ���⿡������ ���� �� ������ ������ ��������, HLS�� ���� �Լ����Դϴ�.
/// 
/// HLS�� ���� ��Ȱȭ�� 3���� �������� ����˴ϴ�.
/// �� ������ HLS ��Ʈ�� ���� �Ʒ� �Լ��� HLS_histogramEqualizationCUDA���� ����˴ϴ�.
/// 
/// 1. HLS������ ������ �޾ƿ���.
/// 2. �޾ƿ� L�����Ϳ� ���� ������׷� ��Ȱȭ�� ����
/// 3. �� ����� ����.
/// 
/// �߿��� �Լ��� ���, //�� ó���Ǹ�, �߿��� �Լ��� ��� ///�� Ȯ���Ͻ� �� �����̴ϴ�.
/// *********************************************************************************
/// </summary>



// HLS�� ���� ������׷��� ����� �մϴ�. ������׷��� 'L�� ��' ��° �ε����� 1 ���մϴ�.
__global__ void HLS_histogramEqualization(const uchar* input, uchar* output, int* hist, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // �� �ȼ��� ����
    if (x < width && y < height)
    {
        int index = (y * width + x);
        // Compute histogram
        // input�� L�� ���� �޾ƿ� �̿� ���� 
        atomicAdd(&hist[input[index]], 1);
    }
}

/// HLS�� L�� ���� ������׷� ��Ȱȭ�� �մϴ�. 
void HLS_histogramNormalize(int* hist, int* cdfNormalized, int totalPixels) {
    //������׷� ����ȭ
    std::vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255);

    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + hist[i];
        cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255);
    }
}

/// HLS�� ���� L�� ��Ȱȭ�� �����͸� ������� output�� ���� �����ɴϴ�.
__global__ void HLS_Equalization(const uchar* input, uchar* output, int* norhist, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x);
        // output�� ��Ȱȭ�� ���� �ֱ� (uchar)�� static_cast ��.
        output[index] = static_cast<uchar>(norhist[input[index]]);
    }
}

/// ���� �Լ���, �޾ƿ� input_image�� ���� ������׷� ��Ȱȭ�� �����ϰ�, �� ����� output_image�� �ֽ��ϴ�.
/// 1. RGB������ ������ HLS������ �������� �ٲٱ�.
/// 2. HLS �����͸� Split �Ͽ� L(Lightness)���� ������ �̿� ���� ��Ȱȭ�� �����մϴ�.
/// 3. HLS�����͸� �ٽ� ��ģ ��, �̸� RGB �������� ��ȯ�Ͽ� ������� �������ϴ�.
void HLS_histogramEqualizationCUDA(const cv::Mat& input_image, cv::Mat& output_image)
{
    int width = input_image.cols;
    int height = input_image.rows;
    int channels = input_image.channels();

    // ������׷� ����
    int hist[256];
    int norHist[256];

    // GPU ������׷� ����
    int* d_hist;
    int* d_norHist;

    cudaMalloc(&d_hist, sizeof(int) * 256);
    cudaMalloc(&d_norHist, sizeof(float) * 256);
    cudaMemset(&d_hist, 0, sizeof(int) * 256);
    cudaMemset(&d_norHist, 0.0f, sizeof(float) * 256);

    // GPU �޸� ����
    uchar* d_input_image;
    uchar* d_output_image;

    cudaMalloc((void**)&d_input_image, input_image.total());
    cudaMalloc((void**)&d_output_image, input_image.total());
    cudaMemcpy(d_input_image, input_image.data, input_image.total(), cudaMemcpyHostToDevice);

    // �׸���� �� ����.
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // HLS�� ���� ������׷��� ����� �մϴ�. ������׷��� 'L�� ��' ��° �ε����� 1 ���մϴ�.
    HLS_histogramEqualization << <gridSize, blockSize >> > (d_input_image, d_output_image, d_hist, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(hist, d_hist, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    /// HLS�� L�� ���� ������׷� ��Ȱȭ�� �մϴ�. �̴� norhist�� ����˴ϴ�.
    HLS_histogramNormalize(&hist[0], norHist, width * height);

    // norhist�� ����Ǿ��� ���� d_norHist�� �÷� �����͸� ����� �� �ֵ��� �մϴ�.
    cudaMemcpy(d_norHist, norHist, sizeof(int) * 256, cudaMemcpyHostToDevice);

    /// HLS�� ���� L�� ��Ȱȭ�� �����͸� ������� output�� ���� �����ɴϴ�.
    HLS_Equalization << < gridSize, blockSize >> > (d_input_image, d_output_image, d_norHist, width, height, channels);
    cudaDeviceSynchronize();

    // ����� outputimage�� �Ӵϴ�.
    cudaMemcpy(output_image.data, d_output_image, output_image.total(), cudaMemcpyDeviceToHost);

    // �޸� free
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    cudaFree(d_norHist);
    cudaFree(d_hist);
}

/// <summary>
/// *********************************************************************************
/// 
/// ���⿡������ ���� �� ������ ������ ��������, RGB�� ���� �Լ����Դϴ�.
/// 
/// RGB�� ���� ��Ȱȭ�� 3���� �������� ����˴ϴ�.
/// �� ������ RGB ��Ʈ�� ���� �Ʒ� �Լ��� RGB_histogramEqualizationCUDA���� ����˴ϴ�.
/// 
/// 1. RGB������ ������ �޾ƿ���.
/// 2. �޾ƿ� RGB ������ ���� ������׷� ��Ȱȭ�� ����
/// 3. �� ����� ����.
/// 
/// �߿��� �Լ��� ���, //�� ó���Ǹ�, �߿��� �Լ��� ��� ///�� Ȯ���Ͻ� �� �����̴ϴ�.
/// *********************************************************************************
/// </summary>

// RGB�� ���� ������׷��� ����� �մϴ�. ������׷��� 'RGB ������ ��' ��° �ε����� 1 ���մϴ�.
__global__ void RGB_histogramEqualization(const uchar* input, uchar* output, int* hist, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * channels;

        // ������׷��� �ε��� 1�� ���ϱ�.
        atomicAdd(&hist[input[index]], 1);
        atomicAdd(&hist[input[index + 1] + 256], 1);
        atomicAdd(&hist[input[index + 2] + 512], 1);
    }
}

/// RGB�� RGB�� ���� ������׷� ��Ȱȭ�� �մϴ�. 
void RGB_histogramNormalize(int* hist, int* cdfNormalized, int totalPixels) {
    //������׷� ��Ȱȭ
    std::vector<int> cdf(256, 0);
    cdf[0] = hist[0];
    cdfNormalized[0] = round((cdf[0] / (float)totalPixels) * 255.0f);

    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
        cdfNormalized[i] = round((cdf[i] / (float)totalPixels) * 255.0f);
    }
}

/// RGB�� ���� ��Ȱȭ�� ������(norhist)�� ������� output�� ���� �����ɴϴ�.
__global__ void RGB_Equalization(const uchar* input, uchar* output, int* norhist, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // �� ��ǥ�� ����
    if (x < width && y < height)
    {
        int index = (y * width + x) * channels;

        output[index] = static_cast<uchar>(norhist[input[index]]);
        output[index + 1] = static_cast<uchar>(norhist[input[index + 1] + 256]);
        output[index + 2] = static_cast<uchar>(norhist[input[index + 2] + 512]);
    }
}


/// ���� �Լ���, �޾ƿ� input_image�� ���� ������׷� ��Ȱȭ�� �����ϰ�, �� ����� output_image�� �ֽ��ϴ�.
/// 1. RGB������ ������ �޾ƿ���.
/// 2. �޾ƿ� RGB ������ ���� ������׷� ��Ȱȭ�� ����
/// 3. �� ����� ����.
void RGB_histogramEqualizationCUDA(const cv::Mat& input_image, cv::Mat& output_image)
{
    int width = input_image.cols;
    int height = input_image.rows;
    int channels = input_image.channels();

    // ������׷� ����
    int hist[768];
    int norHist[768];

    // GPU ������׷� ����
    int* d_hist;
    int* d_norHist;

    cudaMalloc(&d_hist, sizeof(int) * 768);
    cudaMemset(&d_hist, 0, sizeof(int) * 768);
    cudaMalloc(&d_norHist, sizeof(int) * 768);

    // GPU �޸� ����.
    uchar* d_input_image;
    uchar* d_output_image;

    cudaMalloc((void**)&d_input_image, input_image.total() * channels);
    cudaMalloc((void**)&d_output_image, input_image.total() * channels);
    cudaMemcpy(d_input_image, input_image.data, input_image.total() * channels, cudaMemcpyHostToDevice);

    // ��ϰ� �׸��� ����.
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // RGB�� ���� ������׷��� ����� �մϴ�. ������׷��� 'RGB ������ ��' ��° �ε����� 1 ���մϴ�.
    RGB_histogramEqualization << <gridSize, blockSize >> > (d_input_image, d_output_image, d_hist, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(hist, d_hist, sizeof(int) * 768, cudaMemcpyDeviceToHost);

    /// RGB�� ���� ������׷� ��Ȱȭ�� �մϴ�. �̴� norhist�� ����˴ϴ�.
    RGB_histogramNormalize(hist, norHist, width * height);
    RGB_histogramNormalize(&hist[256], &norHist[256], width * height);
    RGB_histogramNormalize(&hist[512], &norHist[512], width * height);

    // norhist�� ����Ǿ��� ���� d_norHist�� �÷� �����͸� ����� �� �ֵ��� �մϴ�.
    cudaMemcpy(d_norHist, norHist, sizeof(int) * 768, cudaMemcpyHostToDevice);

    /// RGB�� ��Ȱȭ�� �����͸� ������� output�� ���� �����ɴϴ�.
    RGB_Equalization << < gridSize, blockSize >> > (d_input_image, d_output_image, d_norHist, width, height, channels);
    cudaDeviceSynchronize();

    // ����� outputimage�� �Ӵϴ�.
    cudaMemcpy(output_image.data, d_output_image, output_image.total() * channels, cudaMemcpyDeviceToHost);

    // �޸� Free
    cudaFree(d_input_image);
    cudaFree(d_output_image);
    cudaFree(d_norHist);
    cudaFree(d_hist);
}


/// <summary>
/// *********************************************************************************
/// 
/// ���⿡������ ���� �� ������ ������ ��������, ��� �κ��� ��ġ�� �κ��Դϴ�.
/// 
/// �̴� ���� HLS�� RGB�� ���������� ����Ǹ�, RGB�� ��� �߰� ������ ������ 
/// HLS�� ��� RGB -> HLS / HLS -> RGB ������ �߰��� ���մϴ�.
/// 
/// ���������� ������� RGB�� HLS�� �ݹݾ� ���� �����ϰ� �˴ϴ�. (���� �����ؼ� �� ���� ��� ������ �� ����)
/// 
/// �߿��� �Լ��� ���, //�� ó���Ǹ�, �߿��� �Լ��� ��� ///�� Ȯ���Ͻ� �� �����̴ϴ�.
/// *********************************************************************************
/// </summary>


// float���鿡 ���� mod���� ���ϴ� �Լ�.
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
/// �־��� uchar* �������� �־��� HSL ������(input)�� RGB�� ��ȯ�� output�� �����ϴ� �Լ�.
// ���� 1 https://www.rapidtables.com/convert/color/hsl-to-rgb.html
// ���� 2 http://dystopiancode.blogspot.com/2012/06/hsl-rgb-conversion-algorithms-in-c.html (���� ������)
// ���� 3 https://codegolf.stackexchange.com/questions/150250/hsl-to-rgb-values (�������� ���� ���ذ� ������ ���� ���� ���ذ� ����)
__global__ void ConvertHLSToRGB(const uchar* input, uchar* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size = width * height;

    // �� �ȼ��� ���� ������ ����˴ϴ�.
    if (x < width && y < height) {

        int index = (y * width + x) * channels;

        float r, g, b;

        // ������ 0~1���� ������ hue�� ���� 0������ 360������ ���� �������� ����.
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

/// �־��� uchar* �������� �־��� RGB ������(input)�� HLS�� ��ȯ�� output�� �����ϴ� �Լ�. 
// ���� 1 : https://stackoverflow.com/questions/39118528/rgb-to-hsl-conversion
__global__ void ConvertRGBToHLS(const uchar* input, uchar* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size = width * height;

    float lightness;
    float saturation;
    float hue;

    // �� �ȼ��� ���� ����.
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

        // ����� ���� ����.
        output[index] = static_cast<uchar>(hue * 255);
        output[index + 1] = static_cast<uchar>(lightness * 255);
        output[index + 2] = static_cast<uchar>(saturation * 255);
    }
}


/// ���� �Լ��� RGB�� ������� input1�� ���� alpha�� ����ġ��, input2�� ���� beta�� ����ġ�� ���� ���� ����� output�� �����ϴ� Ŀ���Դϴ�.
__global__ void CustomWeight(const uchar* input1, double alpha, const uchar* input2, double beta, uchar* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * channels;

        // ���� RGB�� �����͸� (BGR) ������ �°� ��� static_cast ��ŵ�ϴ�.
        output[index] = static_cast<uchar>((alpha * input1[index]) + (beta * input2[index]));
        output[index + 1] = static_cast<uchar>((alpha * input1[index + 1]) + (beta * input2[index + 1]));
        output[index + 2] = static_cast<uchar>((alpha * input1[index + 2]) + (beta * input2[index + 2]));
    }
}

/// ������ �־��� 2���� �̹����� �޾� �̸� alpha�� beta�� ��� ���� ����� ���� �Լ��Դϴ�.
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

    /// ��ġ�� ����
    CustomWeight << <gridSize, blockSize >> > (d_input_image1, alpha, d_input_image2, beta, d_output_image, width, height, channels);

    cudaDeviceSynchronize();
    cudaMemcpy(output_image.data, d_output_image, output_image.total() * channels, cudaMemcpyDeviceToHost);

    cudaFree(d_input_image1);
    cudaFree(d_input_image2);
    cudaFree(d_output_image);

}

/// ���� �ڵ�� �� �������� ���� ���� RGB���� ��Ȱȭ�Ǵ� �Լ��Դϴ�.
void RGBGetWeightedImage(const Mat& input_image, Mat& output_image)
{
    RGB_histogramEqualizationCUDA(input_image, output_image);
}

/// ���� �ڵ�� �� �������� ���� ���� RGB�� HSL���� ���� ��Ȱȭ�Ǵ� �Լ��Դϴ�.
void RGBHSLGetWeightedImage(const Mat& input_image, Mat& output_image)
{
    /// RGB part
    cv::Mat RGB_output_image(input_image.size(), input_image.type());
    RGB_histogramEqualizationCUDA(input_image, RGB_output_image);


    /// HSL part
    cv::Mat hslImage(input_image.size(), input_image.type());
    cv::Mat out(input_image.size(), input_image.type());
    int imageType = input_image.type();

    /*cvtColor �߰��� �κ�*/
    int width = input_image.cols;
    int height = input_image.rows;
    int channel = input_image.channels();

    uchar* d_input_image;
    uchar* d_output_image;
    cudaMalloc((void**)&d_input_image, input_image.total() * channel);
    cudaMalloc((void**)&d_output_image, input_image.total() * channel);

    cudaMemcpy(d_input_image, input_image.data, input_image.total() * channel, cudaMemcpyHostToDevice);

    // ��ϰ� �׸��� ����.
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    /// RGB�����͸� HLS�� ��ȯ
    ConvertRGBToHLS << <gridSize, blockSize >> > (d_input_image, d_output_image, width, height, channel);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data, d_output_image, input_image.total() * channel, cudaMemcpyDeviceToHost);

    hslImage.data = out.data;

    //cv::cvtColor(input_image, hslImage, cv::COLOR_BGR2HLS);
    // ��ȯ�� �����͸� hslImage�� ����.
    // hsl�� ��ȯ�� �����͸� �ɰ��� h, s, l ������ ���� Mat�� ����.

    std::vector<cv::Mat> channels(3);
    cv::split(hslImage, channels);

    /// ��ȯ�� HLS�� ���� L���� ���� ��Ȱȭ ����.
    HLS_histogramEqualizationCUDA(channels[1], channels[1]);


    // L���� ��Ȱȭ�Ǿ����Ƿ�, �ٽ� HLS�� Merge.
    cv::merge(channels, hslImage);

    cv::Mat HSL_output_image(input_image.size(), input_image.type());
    cudaMemcpy(d_input_image, hslImage.data, input_image.total() * channel, cudaMemcpyHostToDevice);

    /// ��Ȱȭ�� HLS �����͸� �ٽ� RGB�� ��ȯ
    ConvertHLSToRGB << <gridSize, blockSize >> > (d_input_image, d_output_image, width, height, channel);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data, d_output_image, input_image.total() * channel, cudaMemcpyDeviceToHost);


    // ��Ȱȭ�� �����Ͱ� RGB�� ��ȯ�Ȱ��� ����.
    HSL_output_image.data = out.data;

    //cv::cvtColor(hslImage, HSL_output_image, cv::COLOR_HLS2BGR);

    /// �� ��� (RGB ��Ȱȭ ��� / HLS ��Ȱȭ ���) ���� ��ġ��
    CustomAddWeighted_CUDA(HSL_output_image, 0.5, RGB_output_image, 0.5, output_image);
}

/// ���� �ڵ�� �� �������� ���� ���� RGB�� HSL���� ���� ��Ȱȭ�Ǵ� �Լ��Դϴ�.
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

    /// ���� ��ġ��
    CustomAddWeighted_CUDA(HSL_output_image, 0.5, RGB_output_image, 0.5, output_image);
}


int main(int argc, char* argv[]) {
    // ������ ���� ����
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
        std::cout << "������ ������ �� �� �����ϴ�." << std::endl;
        return -1;
    }

    // �������� �Ӽ� ��������
    int frame_width = static_cast<int>(capRGBCUDA.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(capRGBCUDA.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capRGBCUDA.get(cv::CAP_PROP_FPS);
    int num_frames = static_cast<int>(capRGBCUDA.get(cv::CAP_PROP_FRAME_COUNT));

    // ���� ������ ����
    cv::VideoWriter outputSerial("outputSerial.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    cv::VideoWriter outputParallel("outputParallel.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    cv::VideoWriter outputRGBCUDA("outputRGBCUDA.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    cv::VideoWriter outputBothCUDA("outputHSLRGBCUDA.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    cv::VideoWriter outputSlowCUDA("outputSlowCUDA.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));

    int i = 0;

    /// Serial (�׳� CPU ����)
    cv::Mat frameSerial, frame_hist_equalized_Serial;
    i = 0;
    while (true) {
        capSerial >> frameSerial;
        if (frameSerial.empty())
            break;

        frame_hist_equalized_Serial.copySize(frameSerial);

        timer.onTimer(2);
        // �� ä�ο� ���� ������׷� ��Ȱȭ ����
        pre::histogramEqualizationSerial(frameSerial, frame_hist_equalized_Serial);
        timer.offTimer(2);
        std::cout << "Serial - " << i++ << "��° ������" << "\n";

        // ��� �������� ���� �����Ϳ� ����
        outputSerial.write(frame_hist_equalized_Serial);
    }


    /// Parallel (Open MP ����)

    cv::Mat frameParallel, frame_hist_equalized_Parallel;
    i = 0;
    while (true) {
        capParallel >> frameParallel;
        if (frameParallel.empty())
            break;

        frame_hist_equalized_Parallel.copySize(frameParallel);

        timer.onTimer(3);
        // �� ä�ο� ���� ������׷� ��Ȱȭ ����
        pre::HistogramEqualizationOpenMP(frameParallel, frame_hist_equalized_Parallel);
        timer.offTimer(3);
        std::cout << "Parallel - " << i++ << "��° ������" << "\n";

        // ��� �������� ���� �����Ϳ� ����
        outputParallel.write(frame_hist_equalized_Parallel);
    }
    

    /// ���� CUDA ����
    i = 0;
    cv::Mat frameSlowCUDA, frame_hist_equalized_Slow_CUDA;
    while (true) {
        capSlowCUDA >> frameSlowCUDA;
        if (frameSlowCUDA.empty())
            break;
        // ����̽����� ȣ��Ʈ�� ������ ����
        std::vector<cv::Mat> channels;
        cv::split(frameSlowCUDA, channels);

        // �� ä�ο� ���� ������׷� ��Ȱȭ ����
        timer.onTimer(4);
        RGBHSLGetWeightedImage_CV(frameSlowCUDA, frame_hist_equalized_Slow_CUDA);
        timer.offTimer(4);
        std::cout << "CV Converting CUDA - " << i++ << "��° ������" << "\n";

        // ��� �������� ���� �����Ϳ� ����
        outputSlowCUDA.write(frame_hist_equalized_Slow_CUDA);
    }

    /// RGB CUDA ����
    i = 0;
    cv::Mat frameRGBCUDA, frame_hist_equalized_RGB_CUDA;
    while (true) {
        capRGBCUDA >> frameRGBCUDA;
        if (frameRGBCUDA.empty())
            break;
        // ����̽����� ȣ��Ʈ�� ������ ����
        std::vector<cv::Mat> channels;
        cv::split(frameRGBCUDA, channels);
        frame_hist_equalized_RGB_CUDA = Mat(frameRGBCUDA.size(), frameRGBCUDA.type());
        // �� ä�ο� ���� ������׷� ��Ȱȭ ����
        timer.onTimer(1);
        RGBGetWeightedImage(frameRGBCUDA, frame_hist_equalized_RGB_CUDA);
        timer.offTimer(1);
        std::cout << "RGB CUDA - " << i++ << "��° ������" << "\n";

        // ��� �������� ���� �����Ϳ� ����
        outputRGBCUDA.write(frame_hist_equalized_RGB_CUDA);
    }

    /// HLS / RGB �̿��� ����
    i = 0;
    cv::Mat frameBothCUDA, frame_hist_equalized_Both_CUDA;
    while (true) {
        capBothCUDA >> frameBothCUDA;
        if (frameBothCUDA.empty())
            break;
        // ����̽����� ȣ��Ʈ�� ������ ����
        std::vector<cv::Mat> channels;
        cv::split(frameBothCUDA, channels);

        // �� ä�ο� ���� ������׷� ��Ȱȭ ����
        timer.onTimer(5);
        RGBHSLGetWeightedImage(frameBothCUDA, frame_hist_equalized_Both_CUDA);
        timer.offTimer(5);
        std::cout << "HLS / RGB CUDA - " << i++ << "��° ������" << "\n";

        // ��� �������� ���� �����Ϳ� ����
        outputBothCUDA.write(frame_hist_equalized_Both_CUDA);
    }


    // ���� ���ϰ� ���� ������ �ݱ�
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


    std::cout << "������׷� ��Ȱȭ�� �������� output.mp4 ���Ϸ� ����Ǿ����ϴ�." << std::endl;
    std::cout << "Serial ���� ��ü ������" << i << "���� �ð� : " << timer.getTimer_ms(2) << "ms" << std::endl;
    std::cout << "Serial ���� �� �����Ӵ� ��� �ð� : " << timer.getTimer_ms(2) / i << "ms" << std::endl;
    std::cout << "Parallel ���� ��ü ������" << i << "���� �ð� : " << timer.getTimer_ms(3) << "ms" << std::endl;
    std::cout << "Parallel ���� �� �����Ӵ� ��� �ð� : " << timer.getTimer_ms(3) / i << "ms" << std::endl;
    std::cout << "CUDA CV Converting ���� ��ü ������ : " << i << " ���� �ð� : " << timer.getTimer_ms(4) << "ms" << std::endl;
    std::cout << "CUDA CV Converting ���� �� �����Ӵ� ��� �ð� : " << timer.getTimer_ms(4) / i << "ms" << std::endl;
    std::cout << "CUDA RGB ���� ��ü ������ : " << i << " ���� �ð� : " << timer.getTimer_ms(1) << "ms" << std::endl;
    std::cout << "CUDA RGB ���� �� �����Ӵ� ��� �ð� : " << timer.getTimer_ms(1) / i << "ms" << std::endl;
    std::cout << "CUDA HSL / RGB ���� ��ü ������ : " << i << " ���� �ð� : " << timer.getTimer_ms(5) << "ms" << std::endl;
    std::cout << "CUDA HSL / RGB ���� �� �����Ӵ� ��� �ð� : " << timer.getTimer_ms(5) / i << "ms" << std::endl;
    return 0;
}