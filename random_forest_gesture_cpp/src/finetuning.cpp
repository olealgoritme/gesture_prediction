// finetuning.cpp - Fine-tune the gesture recognition model with new data
#include <dlib/matrix.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <string>

// Define the sample type
using sample_type = dlib::matrix<double, 15, 1>;
using label_type = double;

int main() {
    // Check if new data file exists
    std::string new_data_path = "../data/new_gesture_data.csv";
    std::ifstream new_data_file(new_data_path);
    if (!new_data_file.is_open()) {
        std::cerr << "New data file not found: " << new_data_path << std::endl;
        return 1;
    }

    // Load new dataset
    std::vector<sample_type> samples;
    std::vector<label_type> labels;

    std::string line;
    std::getline(new_data_file, line);  // Skip header

    while (std::getline(new_data_file, line)) {
        std::stringstream ss(line);
        sample_type sample;
        label_type label;
        std::string value;
        int i = 0;

        while (std::getline(ss, value, ',')) {
            if (i < 15) {
                sample(i) = std::stod(value);
            } else {
                label = (value == "intentional") ? 1 : 0;
            }
            ++i;
        }
        samples.push_back(sample);
        labels.push_back(label);
    }

    // Convert to OpenCV format
    cv::Mat new_samples(samples.size(), 15, CV_32F);
    cv::Mat new_labels(samples.size(), 1, CV_32S);
    for (size_t i = 0; i < samples.size(); ++i) {
        for (int j = 0; j < 15; ++j) {
            new_samples.at<float>(i, j) = samples[i](j);
        }
        new_labels.at<int>(i, 0) = labels[i];
    }

    // Split the data
    cv::Mat train_data_new, test_data_new;
    cv::Mat train_labels_new, test_labels_new;
    int split_index = static_cast<int>(samples.size() * 0.8);
    new_samples.rowRange(0, split_index).copyTo(train_data_new);
    new_labels.rowRange(0, split_index).copyTo(train_labels_new);
    new_samples.rowRange(split_index, samples.size()).copyTo(test_data_new);
    new_labels.rowRange(split_index, samples.size()).copyTo(test_labels_new);

    // Load the saved model checkpoint
    std::string model_path = "../models/gesture_model.xml";
    cv::Ptr<cv::ml::RTrees> forest = cv::ml::RTrees::load(model_path);
    if (forest.empty()) {
        std::cerr << "Failed to load model from: " << model_path << std::endl;
        return 1;
    }

    // Update the model with new data
    forest->train(train_data_new, cv::ml::ROW_SAMPLE, train_labels_new);

    // Save the updated model checkpoint
    std::string updated_checkpoint_path = "../models/finetuned_gesture_model.xml";
    forest->save(updated_checkpoint_path);

    // Make predictions
    cv::Mat predictions_new;
    forest->predict(test_data_new, predictions_new);

    // Evaluate the model
    int correct_new = 0;
    for (int i = 0; i < predictions_new.rows; ++i) {
        if (predictions_new.at<float>(i, 0) == test_labels_new.at<int>(i, 0)) {
            ++correct_new;
        }
    }
    double accuracy_new = static_cast<double>(correct_new) / test_labels_new.rows;
    std::cout << "New Accuracy: " << accuracy_new << std::endl;

    return 0;
}
