#include "model.h"
#include <fstream>
#include <print>
#include <random>
#include <chrono>
int main(int argc, char* argv[]){
    if (argc < 7) {
    std::cerr << "Usage: ./main <n_estimators> <max_depth> <learning_rate> <gamma> <lambda> <split_ratio>\n";
    return 1;
    }
    int n_estimators = std::stoi(argv[1]);
    int max_depth = std::stoi(argv[2]);
    double learning_rate = std::stod(argv[3]);
    double gamma = std::stod(argv[4]);
    double lambda = std::stod(argv[5]);
    double split_ratio = std::stod(argv[6]);
    XGBoost model{ n_estimators, max_depth, learning_rate, gamma, lambda};
    std::vector<std::vector<double>> dataset;
    std::random_device dev;
    std::mt19937 seed{dev()};

    std::ifstream fileHandle ("/Users/anthai/c++xgboost/src/housing_num.csv");
    if(!fileHandle){
        std::cerr<< "Failed to open housing_num.csv\n";
        return 1;
    };
    std::string line;

    std::getline(fileHandle, line);
    while (getline(fileHandle, line)){
        std::vector<double> row;
        std::stringstream ss(line);
        std::string token;
        while(getline(ss, token, ',')){
            row.push_back(std::stod(token));
        };
        dataset.push_back(row);
    };
    
    
    std::shuffle(dataset.begin(), dataset.end(), seed);
    
   


    auto split_index = static_cast<size_t>((double)dataset.size() * split_ratio);
    auto train_data = std::vector<std::vector<double>>(dataset.begin(), dataset.begin()+split_index);
    std::vector<double> train_label, test_label;
    train_label.reserve(train_data.size());
    for (auto& row: train_data){
        if(row.size()>=2){
        train_label.push_back(*(row.end()-2));
        row.erase(row.end()-2);
        }
    };

    auto test_data = std::vector<std::vector<double>>(dataset.begin()+ split_index, dataset.end());
    test_label.reserve(test_data.size());
    for (auto &row: test_data){
        if(row.size()>=2){
            test_label.push_back(*(row.end()-2));
            row.erase(row.end()-2);
        }
    };
    auto start = std::chrono::high_resolution_clock::now();
    model.train(train_data, train_label);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    std::println("Training time is: {}", duration);
    model.test(test_data, test_label);
    model.saveModel("../savemodel.bin");
    XGBoost test_model{};
    test_model.loadModel("../savemodel.bin");
    test_model.test(test_data, test_label);
    return 0;
    
}