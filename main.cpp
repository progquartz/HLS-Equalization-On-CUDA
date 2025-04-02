#include <opencv2/opencv.hpp>
/*
int main() {
    // 동영상 파일 열기
    cv::VideoCapture cap("sea.mp4");
    if (!cap.isOpened()) {
        std::cout << "동영상 파일을 열 수 없습니다." << std::endl;
        return -1;
    }

    // 동영상의 속성 가져오기
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int num_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // 비디오 라이터 생성
    cv::VideoWriter output("output.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame, frame_hist_equalized;

    // 각 프레임에 대해 히스토그램 평활화 수행
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // 채널 분리
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);

        // 각 채널에 대해 히스토그램 평활화 수행
        for (int i = 0; i < channels.size(); ++i) {
            cv::equalizeHist(channels[i], channels[i]);
        }

        // 채널을 8비트로 변환하여 병합
        cv::merge(channels, frame_hist_equalized);

        // 결과 프레임을 비디오 라이터에 저장
        output.write(frame_hist_equalized);
    }

    // 비디오 파일과 비디오 라이터 닫기
    cap.release();
    output.release();

    std::cout << "히스토그램 평활화된 동영상이 output.mp4 파일로 저장되었습니다." << std::endl;

    return 0;
}
*/
