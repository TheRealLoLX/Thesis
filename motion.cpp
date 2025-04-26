#include <opencv2/opencv.hpp>
#include <mediapipe/framework/calculator_graph.h>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <mediapipe/framework/formats/classification.pb.h>
#include <mediapipe/framework/port/status.h>
#include <mediapipe/framework/port/parse_text_proto.h>
#include <mediapipe/framework/port/opencv_core_inc.h>
#include <mediapipe/framework/port/opencv_highgui_inc.h>
#include <mediapipe/framework/port/opencv_video_inc.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_mixer.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <deque>

using namespace cv;
using namespace std;

// Parameters
const float pinch_threshold = 0.03f;
const float swipe_threshold = 20.0f;  // lower because we smooth it
const float scale_threshold = 0.1f;   // 10% relative change triggers spiral
const int buffer_size = 5;             // number of frames for smoothing

// Buffers for smooth swipe detection
deque<float> movement_x_buffer;
deque<float> movement_y_buffer;

// Previous values
float previous_distance = -1.0f;
float prev_x_right = -1.0f;
float prev_y_right = -1.0f;

// Sounds
Mix_Chunk* swipe_right_sound;
Mix_Chunk* swipe_left_sound;
Mix_Chunk* swipe_up_sound;
Mix_Chunk* swipe_down_sound;

// Helper Functions
float calculate_distance(float x1, float y1, float x2, float y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

pair<float, float> get_hand_center(const vector<mediapipe::NormalizedLandmark>& landmarks) {
    float sum_x = 0.0f, sum_y = 0.0f;
    for (const auto& lm : landmarks) {
        sum_x += lm.x();
        sum_y += lm.y();
    }
    return {sum_x / landmarks.size(), sum_y / landmarks.size()};
}

bool check_pinch(const vector<mediapipe::NormalizedLandmark>& landmarks) {
    const auto& thumb_tip = landmarks[4];
    const auto& middle_tip = landmarks[12];
    float distance = calculate_distance(thumb_tip.x(), thumb_tip.y(), middle_tip.x(), middle_tip.y());
    return (distance < pinch_threshold && fabs(thumb_tip.y() - middle_tip.y()) < 0.05f);
}

string recognize_gesture(const vector<mediapipe::NormalizedLandmark>& landmarks) {
    vector<int> fingers;
    int tips[5] = {4, 8, 12, 16, 20};
    for (int i = 1; i <= 4; ++i) {
        if (landmarks[tips[i]].y() < landmarks[tips[i] - 2].y())
            fingers.push_back(1);
        else
            fingers.push_back(0);
    }
    bool thumb = landmarks[tips[0]].x() < landmarks[tips[0] - 1].x();
    int sum_fingers = accumulate(fingers.begin(), fingers.end(), 0);

    if (sum_fingers == 0) return "Fist";
    else if (sum_fingers == 4) return "Palm";
    else if (fingers[0] == 1 && fingers[1] == 1 && fingers[2] == 0 && fingers[3] == 0) return "Peace";
    return "Unknown";
}

void update_buffer(deque<float>& buffer, float value) {
    if (buffer.size() >= buffer_size) buffer.pop_front();
    buffer.push_back(value);
}

float average(const deque<float>& buffer) {
    if (buffer.empty()) return 0.0f;
    return accumulate(buffer.begin(), buffer.end(), 0.0f) / buffer.size();
}

int main() {
    // SDL2 mixer initialization
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        cerr << "SDL_Init Error: " << SDL_GetError() << endl;
        return -1;
    }
    if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0) {
        cerr << "Mix_OpenAudio Error: " << Mix_GetError() << endl;
        return -1;
    }

    swipe_right_sound = Mix_LoadWAV("C:/Users/hbeci/PycharmProjects/Thesis/swipe_right.mp3");
    swipe_left_sound = Mix_LoadWAV("C:/Users/hbeci/PycharmProjects/Thesis/swipe_left.mp3");
    swipe_up_sound = Mix_LoadWAV("C:/Users/hbeci/PycharmProjects/Thesis/swipe_up.mp3");
    swipe_down_sound = Mix_LoadWAV("C:/Users/hbeci/PycharmProjects/Thesis/swipe_down.mp3");

    if (!swipe_right_sound || !swipe_left_sound || !swipe_up_sound || !swipe_down_sound) {
        cerr << "Failed to load sound: " << Mix_GetError() << endl;
        return -1;
    }

    // MediaPipe Graph Initialization
    mediapipe::CalculatorGraph graph;
    string graph_config_contents;
    {
        std::ifstream graph_file("hand_tracking_desktop_live.pbtxt");
        graph_config_contents.assign((istreambuf_iterator<char>(graph_file)),
                                     istreambuf_iterator<char>());
    }
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_config_contents);
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Failed to open camera." << endl;
        return -1;
    }

    ASSIGN_OR_RETURN(auto poller_landmarks, graph.AddOutputStreamPoller("multi_hand_landmarks"));
    ASSIGN_OR_RETURN(auto poller_handedness, graph.AddOutputStreamPoller("multi_handedness"));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    namedWindow("Hand Gesture Detection", WINDOW_NORMAL);

    while (true) {
        Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty()) break;

        flip(camera_frame_raw, camera_frame_raw, 1);
        Mat camera_frame;
        cvtColor(camera_frame_raw, camera_frame, COLOR_BGR2RGB);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            "input_video", mediapipe::Adopt(input_frame.release())
                           .At(mediapipe::Timestamp::NextAllowedInStream())));

        mediapipe::Packet packet_landmarks;
        if (!poller_landmarks.Next(&packet_landmarks)) break;
        auto& output_landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

        mediapipe::Packet packet_handedness;
        poller_handedness.Next(&packet_handedness);
        auto& output_handedness = packet_handedness.Get<std::vector<mediapipe::ClassificationList>>();

        vector<mediapipe::NormalizedLandmark> left_hand, right_hand;
        bool left_pinch = false, right_pinch = false;

        for (size_t i = 0; i < output_landmarks.size(); ++i) {
            const auto& landmark_list = output_landmarks[i];
            const auto& handedness = output_handedness[i];

            string label = handedness.classification(0).label();
            vector<mediapipe::NormalizedLandmark> landmarks;
            for (int j = 0; j < landmark_list.landmark_size(); ++j)
                landmarks.push_back(landmark_list.landmark(j));

            if (label == "Left") {
                left_hand = landmarks;
                left_pinch = check_pinch(landmarks);
                if (!left_pinch) {
                    string gesture = recognize_gesture(landmarks);
                    putText(camera_frame_raw, "Left: " + gesture, Point(30, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                }
            }
            if (label == "Right") {
                right_hand = landmarks;
                right_pinch = check_pinch(landmarks);

                auto [hand_x, hand_y] = get_hand_center(landmarks);
                hand_x *= camera_frame_raw.cols;
                hand_y *= camera_frame_raw.rows;

                if (!right_pinch) {
                    if (prev_x_right >= 0 && prev_y_right >= 0) {
                        float move_x = hand_x - prev_x_right;
                        float move_y = hand_y - prev_y_right;

                        update_buffer(movement_x_buffer, move_x);
                        update_buffer(movement_y_buffer, move_y);

                        if (abs(average(movement_x_buffer)) > swipe_threshold) {
                            if (average(movement_x_buffer) > 0) {
                                cout << "Smooth Swipe Right!" << endl;
                                Mix_PlayChannel(-1, swipe_right_sound, 0);
                            } else {
                                cout << "Smooth Swipe Left!" << endl;
                                Mix_PlayChannel(-1, swipe_left_sound, 0);
                            }
                            movement_x_buffer.clear();
                        }
                        if (abs(average(movement_y_buffer)) > swipe_threshold) {
                            if (average(movement_y_buffer) > 0) {
                                cout << "Smooth Swipe Down!" << endl;
                                Mix_PlayChannel(-1, swipe_down_sound, 0);
                            } else {
                                cout << "Smooth Swipe Up!" << endl;
                                Mix_PlayChannel(-1, swipe_up_sound, 0);
                            }
                            movement_y_buffer.clear();
                        }
                    }
                    prev_x_right = hand_x;
                    prev_y_right = hand_y;
                }
            }
        }

        if (!left_hand.empty() && !right_hand.empty() && left_pinch && right_pinch) {
            float current_distance = calculate_distance(
                get_hand_center(left_hand).first, get_hand_center(left_hand).second,
                get_hand_center(right_hand).first, get_hand_center(right_hand).second
            );
            if (previous_distance > 0) {
                float scale_change = (current_distance - previous_distance) / previous_distance;
                if (fabs(scale_change) > scale_threshold) {
                    if (scale_change < 0)
                        cout << "Smooth Left Spiral!" << endl;
                    else
                        cout << "Smooth Right Spiral!" << endl;
                }
            }
            previous_distance = current_distance;
        }

        imshow("Hand Gesture Detection", camera_frame_raw);
        if (waitKey(5) == 'q') break;
    }

    Mix_CloseAudio();
    SDL_Quit();
    capture.release();
    destroyAllWindows();

    return 0;
}
