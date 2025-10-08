#pragma once
#include <vector>
#include <memory>
struct TreeNode {
    bool is_leaf;
    int feature_index;
    double threshold;
    double value;  
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;

    TreeNode() : is_leaf(false), feature_index(-1), threshold(0.0),
                 value(0.0), left(nullptr), right(nullptr) {};


};