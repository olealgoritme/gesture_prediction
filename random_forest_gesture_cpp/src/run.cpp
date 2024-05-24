// run.cpp - Inference
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <algorithm>

using sample_type = cv::Mat;

sample_type create_sample() {
    // Example gesture data for inference
    double x1_start = 400;
    double y1_start = 300;
    double x1_end = 500;
    double y1_end = 350;
    double pressure1_start = 0.8;
    double pressure1_end = 0.6;
    double x2_start = 450;
    double y2_start = 350;
    double x2_end = 550;
    double y2_end = 400;
    double pressure2_start = 0.6;
    double pressure2_end = 0.4;
    double timestamp1_start = 0.05;
    double timestamp1_end = 0.1;
    double timestamp2_start = 0.05;
    double timestamp2_end = 0.1;

    double timestamp_start = std::min(timestamp1_start, timestamp2_start);
    double timestamp_end = std::max(timestamp1_end, timestamp2_end);

    double duration1 = timestamp1_end - timestamp1_start;
    double duration2 = timestamp2_end - timestamp2_start;
    double velocity1 = std::sqrt(std::pow(x1_end - x1_start, 2) + std::pow(y1_end - y1_start, 2)) / duration1;
    double velocity2 = std::sqrt(std::pow(x2_end - x2_start, 2) + std::pow(y2_end - y2_start, 2)) / duration2;

    if (std::isinf(velocity1)) velocity1 = 0;
    if (std::isinf(velocity2)) velocity2 = 0;

    sample_type sample(1, 15, CV_32F);
    sample.at<float>(0, 0) = x1_start;
    sample.at<float>(0, 1) = y1_start;
    sample.at<float>(0, 2) = x1_end;
    sample.at<float>(0, 3) = y1_end;
    sample.at<float>(0, 4) = pressure1_start;
    sample.at<float>(0, 5) = pressure1_end;
    sample.at<float>(0, 6) = velocity1;
    sample.at<float>(0, 7) = x2_start;
    sample.at<float>(0, 8) = y2_start;
    sample.at<float>(0, 9) = x2_end;
    sample.at<float>(0, 10) = y2_end;
    sample.at<float>(0, 11) = pressure2_start;
    sample.at<float>(0, 12) = pressure2_end;
    sample.at<float>(0, 13) = velocity2;
    sample.at<float>(0, 14) = timestamp_start;
    sample.at<float>(0, 15) = timestamp_end;

    return sample;
}

int main() {
    // Load the model with fallback
    std::ifstream finetuned_model_file("../models/finetuned_gesture_model.xml");
    std::ifstream model_file("../models/gesture_model.xml");

    if (!finetuned_model_file.good() && !model_file.good()) {
        std::cerr << "Neither finetuned_gesture_model.xml nor gesture_model.xml exist." << std::endl;
        return 1;
    }

    // Load the model with fallback
    cv::Ptr<cv::ml::RTrees> forest;
    if (finetuned_model_file.good()) {
        if (forest.empty()) {
          forest = cv::ml::RTrees::load("../models/finetuned_gesture_model.xml");
            std::cerr << "Failed to load finetuned_gesture_model.xml" << std::endl;
            return 1;
        }
        std::cout << "Loaded finetuned_gesture_model.xml" << std::endl;
    } else if (model_file.good()) {
        forest = cv::ml::RTrees::load("../models/gesture_model.xml");
        if (forest.empty()) {
            std::cerr << "Failed to load gesture_model.xml" << std::endl;
            return 1;
        }
        std::cout << "Loaded gesture_model.xml" << std::endl;
    }

    // Create sample for inference
    sample_type sample = create_sample();

    // Perform inference
    float prediction = forest->predict(sample);

    // Map the prediction back to label
    std::string predicted_label = (prediction == 1.0f) ? "intentional" : "accidental";

    std::cout << "Predicted Gesture: " << predicted_label << std::endl;

    return 0;
}
