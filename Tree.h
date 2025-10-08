#ifndef TREE
#define TREE
#include "TreeNode.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "ThreadPool.h"
#include "bestFeatureSplit.h"
#include "execution"
#include "ranges"
#include <cmath>
#include <print>
#include <fstream>
class Tree {
public: 
    Tree(int max_depth, double gamma, double lambda, std::vector<std::vector<double>>& dataset, std::vector<std::vector<int>>& sorted_dataset_indices, std::vector<double>& preComputedG, std::vector<double>& preComputedH, ThreadPool& thread_pool);
    ~Tree() = default;

    void build(const std::vector<int>& data);
    double predict(const std::vector<double>& data) const;
    void save(std::ofstream& out)const;
    void load(std::ifstream& in);
    std::unique_ptr<TreeNode> buildNode(const std::vector<int>& data, int depth);
private:
    int max_depth_;
    double gamma_;
    double lambda_;
    std::vector<std::vector<double>>& dataset_;
    std::vector<std::vector<int>>& sorted_dataset_indices_;
    std::vector<double>& preComputedG_;
    std::vector<double>& preComputedH_;
    ThreadPool& thread_pool_;
    std::unique_ptr<TreeNode> root;
    bestFeatureSplit findBestSplit(const std::vector<int>& data);
    bestFeatureSplit findFeatureSplit(const std::vector<int>& data, int feature);
    double computeGain(double G_left, double H_left, double G_right, double H_right);
    double computeWeight(const std::vector<int>& data);

};
#endif