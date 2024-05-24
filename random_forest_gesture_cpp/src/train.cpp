// train.cpp - Trains a Random Forest classifier for gesture recognition
#include <dlib/matrix.h>
#include <dlib/data_io.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <string>

// Define the sample type
using sample_type = dlib::matrix<double, 15, 1>;
using label_type = double;

int main() {
    // Load the dataset
    std::ifstream data_file("../data/gesture_data.csv");
    if (!data_file.is_open()) {
        std::cerr << "Failed to open the dataset file" << std::endl;
        return 1;
    }

    std::vector<sample_type> samples;
    std::vector<label_type> labels;

    std::string line;
    std::getline(data_file, line);  // Skip header

    while (std::getline(data_file, line)) {
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
    cv::Mat training_samples(samples.size(), 15, CV_32F);
    cv::Mat training_labels(samples.size(), 1, CV_32S);
    for (size_t i = 0; i < samples.size(); ++i) {
        for (int j = 0; j < 15; ++j) {
            training_samples.at<float>(i, j) = samples[i](j);
        }
        training_labels.at<int>(i, 0) = labels[i];
    }

    // Split the data
    cv::Mat train_data, test_data;
    cv::Mat train_labels, test_labels;
    int split_index = static_cast<int>(samples.size() * 0.8);
    training_samples.rowRange(0, split_index).copyTo(train_data);
    training_labels.rowRange(0, split_index).copyTo(train_labels);
    training_samples.rowRange(split_index, samples.size()).copyTo(test_data);
    training_labels.rowRange(split_index, samples.size()).copyTo(test_labels);

    // Train the Random Forest classifier using OpenCV
    cv::Ptr<cv::ml::RTrees> forest = cv::ml::RTrees::create();
    forest->setMaxDepth(10);
    forest->setMinSampleCount(10);
    forest->setRegressionAccuracy(0.0);
    forest->setUseSurrogates(false);
    forest->setMaxCategories(2);
    forest->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 0.01));

    forest->train(train_data, cv::ml::ROW_SAMPLE, train_labels);

    std::string outputModel("../models/gesture_model.xml");
    // Save the model
    forest->save(outputModel);

    // Make predictions
    cv::Mat predictions;
    forest->predict(test_data, predictions);

    // Evaluate the model
    int correct = 0;
    for (int i = 0; i < predictions.rows; ++i) {
        if (predictions.at<float>(i, 0) == test_labels.at<int>(i, 0)) {
            ++correct;
        }
    }
    double accuracy = static_cast<double>(correct) / test_labels.rows;
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "Trained model saved to: " << outputModel << std::endl;

    return 0;
}
