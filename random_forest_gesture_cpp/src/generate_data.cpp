// generate_data.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

struct GestureData {
    int x1_start, y1_start, x1_end, y1_end;
    double pressure1_start, pressure1_end;
    int x2_start, y2_start, x2_end, y2_end;
    double pressure2_start, pressure2_end;
    double timestamp1_start, timestamp1_end;
    double timestamp2_start, timestamp2_end;
    double timestamp_start, timestamp_end;
    std::string label;
};

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist_x1(0, 1024);
    std::uniform_int_distribution<> dist_y1(0, 768);
    std::uniform_int_distribution<> dist_x2(0, 1024);
    std::uniform_int_distribution<> dist_y2(0, 768);
    std::uniform_real_distribution<> dist_pressure(0.1, 1.0);
    std::uniform_real_distribution<> dist_time(0.01, 0.1);
    std::uniform_int_distribution<> dist_label(0, 1);

    const size_t num_samples = 100000;
    std::vector<GestureData> data(num_samples);

    double cumulative_time1 = 0.0;
    double cumulative_time2 = 0.0;

    for (size_t i = 0; i < num_samples; ++i) {
        data[i].x1_start = dist_x1(gen);
        data[i].y1_start = dist_y1(gen);
        data[i].x1_end = dist_x1(gen);
        data[i].y1_end = dist_y1(gen);
        data[i].pressure1_start = dist_pressure(gen);
        data[i].pressure1_end = dist_pressure(gen);
        data[i].x2_start = dist_x2(gen);
        data[i].y2_start = dist_y2(gen);
        data[i].x2_end = dist_x2(gen);
        data[i].y2_end = dist_y2(gen);
        data[i].pressure2_start = dist_pressure(gen);
        data[i].pressure2_end = dist_pressure(gen);
        cumulative_time1 += dist_time(gen);
        cumulative_time2 += dist_time(gen);
        data[i].timestamp1_start = cumulative_time1;
        data[i].timestamp1_end = cumulative_time1 + dist_time(gen);
        data[i].timestamp2_start = cumulative_time2;
        data[i].timestamp2_end = cumulative_time2 + dist_time(gen);
        data[i].timestamp_start = std::min(data[i].timestamp1_start, data[i].timestamp2_start);
        data[i].timestamp_end = std::max(data[i].timestamp1_end, data[i].timestamp2_end);
        data[i].label = dist_label(gen) == 0 ? "intentional" : "accidental";
    }

    std::ofstream file("../data/gesture_data.csv");
    file << "x1_start,y1_start,x1_end,y1_end,pressure1_start,pressure1_end,x2_start,y2_start,x2_end,y2_end,pressure2_start,pressure2_end,timestamp1_start,timestamp1_end,timestamp2_start,timestamp2_end,timestamp_start,timestamp_end,label\n";

    for (const auto& d : data) {
        file << d.x1_start << ',' << d.y1_start << ',' << d.x1_end << ',' << d.y1_end << ','
             << d.pressure1_start << ',' << d.pressure1_end << ','
             << d.x2_start << ',' << d.y2_start << ',' << d.x2_end << ',' << d.y2_end << ','
             << d.pressure2_start << ',' << d.pressure2_end << ','
             << d.timestamp1_start << ',' << d.timestamp1_end << ','
             << d.timestamp2_start << ',' << d.timestamp2_end << ','
             << d.timestamp_start << ',' << d.timestamp_end << ','
             << d.label << '\n';
    }

    std::cout << "Generated data and saved to ../data/gesture_data.csv" << std::endl;

    return 0;
}
