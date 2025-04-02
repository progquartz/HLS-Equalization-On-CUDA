#include <opencv2/opencv.hpp>
/*
int main() {
    // ������ ���� ����
    cv::VideoCapture cap("sea.mp4");
    if (!cap.isOpened()) {
        std::cout << "������ ������ �� �� �����ϴ�." << std::endl;
        return -1;
    }

    // �������� �Ӽ� ��������
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int num_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // ���� ������ ����
    cv::VideoWriter output("output.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame, frame_hist_equalized;

    // �� �����ӿ� ���� ������׷� ��Ȱȭ ����
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        // ä�� �и�
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);

        // �� ä�ο� ���� ������׷� ��Ȱȭ ����
        for (int i = 0; i < channels.size(); ++i) {
            cv::equalizeHist(channels[i], channels[i]);
        }

        // ä���� 8��Ʈ�� ��ȯ�Ͽ� ����
        cv::merge(channels, frame_hist_equalized);

        // ��� �������� ���� �����Ϳ� ����
        output.write(frame_hist_equalized);
    }

    // ���� ���ϰ� ���� ������ �ݱ�
    cap.release();
    output.release();

    std::cout << "������׷� ��Ȱȭ�� �������� output.mp4 ���Ϸ� ����Ǿ����ϴ�." << std::endl;

    return 0;
}
*/
