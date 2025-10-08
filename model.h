#ifndef MODEL
#define MODEL

#include <iostream>
#include <vector>
#include <filesystem>
#include "Tree.h"
#include <print>
class XGBoost {
public:
    XGBoost();
    XGBoost(int n_estimators, int max_depth, double learning_rate, double gamma, double lambda);
    void train(std::vector<std::vector<double>>& train_data, const std::vector<double>& train_label);
    void test(const std::vector<std::vector<double>>& test_data, const std::vector<double>& test_label);
    double predict(const std::vector<double>& row) const;
    void saveModel (const std::filesystem::path& filename) const;
    void loadModel (const std::filesystem::path& filename);
private:
    double base_pred_ = 0.0;
    int n_estimators_ = 1;
    int max_depth_ = 10;
    double learning_rate_ = 0.3;
    double gamma_ = 0.1;
    double lambda_ = 1.0;
    std::vector<std::vector<double>> dummy_dataset_;
    std::vector<std::vector<int>> dummy_sorted_indices_;
    std::vector<double> dummy_G_;
    std::vector<double> dummy_H_;
    ThreadPool thread_pool{std::thread::hardware_concurrency()};
    std::vector<std::unique_ptr<Tree>> trees;


};

#endif