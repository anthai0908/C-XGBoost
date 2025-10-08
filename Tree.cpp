#include "Tree.h"

Tree::Tree(int max_depth, double gamma, double lambda, std::vector<std::vector<double>>& dataset, std::vector<std::vector<int>>& sorted_dataset_indices, std::vector<double>& preComputedG, std::vector<double>& preComputedH, ThreadPool& thread_pool) 
: max_depth_(max_depth), gamma_(gamma), lambda_(lambda), dataset_(dataset), sorted_dataset_indices_(sorted_dataset_indices), preComputedG_(preComputedG),preComputedH_(preComputedH), thread_pool_(thread_pool){
    std::println("Initialising one tree...");
};
void Tree::build(const std::vector<int>& data){
    this->root = this->buildNode(data, 0);
};


std::unique_ptr<TreeNode> Tree::buildNode(const std::vector<int>& data, int depth){
    if(data.empty()){
        return nullptr;
    };
    std::unique_ptr<TreeNode> node = std::make_unique<TreeNode>();
    if(depth>= this->max_depth_){
        node->is_leaf = true;
        node->value = this->computeWeight(data);
        return node;
    }
    /// Order of tuple is (feature index, feature threshold, computed gain)
    bestFeatureSplit best_split = this->findBestSplit(data);  

    
    
    if(best_split.feature_index == -1 || best_split.gain <= 0){
        node -> is_leaf = true;
        node->value = this->computeWeight(data);
        return node;    
    };
    
    node -> feature_index = best_split.feature_index;
    node -> threshold = best_split.feature_threshold;
    std::vector<int> copy_data = data;
    auto mid = std::partition(copy_data.begin(), copy_data.end(), [&](int x){
        return dataset_[x][node->feature_index] <= node->threshold;
    });
    auto left_data = std::vector<int>(copy_data.begin(), mid);
    auto right_data = std::vector<int>(mid, copy_data.end());
    if(left_data.empty() || right_data.empty()){
        node->is_leaf = true;
        node->value = this->computeWeight(data);
        return node;
    }
    auto left_node = this->buildNode(left_data, depth+1);
    auto right_node = this->buildNode(right_data, depth+1);
    node->left = std::move(left_node);
    node->right = std::move(right_node);
    return node;
};


bestFeatureSplit Tree::findBestSplit(const std::vector<int>& data){
    std::vector<std::future<bestFeatureSplit>> feature_fut_vec;
    for (size_t feature = 0; feature < this->dataset_[0].size(); feature++){
        std::future<bestFeatureSplit> feature_fut = thread_pool_.enqueue(&Tree::findFeatureSplit, this, std::ref(data), feature);
        feature_fut_vec.push_back(std::move(feature_fut));
    };
    bestFeatureSplit best_split{};
    double max_gain = -std::numeric_limits<double>::infinity();
    for (auto &fut: feature_fut_vec){
        auto split_candidate = fut.get();
        if (split_candidate.gain >max_gain){
            max_gain = split_candidate.gain;
            best_split = split_candidate;
        };
    };
    return best_split;
};


bestFeatureSplit Tree::findFeatureSplit(
    const std::vector<int>& data,                                      
    int feature){
        bestFeatureSplit result{};
        result.feature_index = -1;
        result.feature_threshold = 0.0;
        result.gain = -std::numeric_limits<double>::infinity();
        if (data.empty()){ 
            return result;
        };
        size_t size = data.size();
        std::vector<int> sorted_feature_indices_mask(this->dataset_.size(), 0);
        for (auto i : data){
            sorted_feature_indices_mask[i] =1;
        };
        std::vector<int> sorted_feature_indices_vec;
        std::vector<double>  sorted_G_vals, sorted_H_vals;
        sorted_feature_indices_vec.reserve(size);
        sorted_G_vals.reserve(size);
        sorted_H_vals.reserve(size);
        for (auto i : this->sorted_dataset_indices_[feature]){
            if (sorted_feature_indices_mask[i] ==1){
                sorted_feature_indices_vec.push_back(i);
                sorted_G_vals.push_back(this->preComputedG_[i]);
                sorted_H_vals.push_back(this->preComputedH_[i]);
            };
        };
        
        if(sorted_feature_indices_vec.size() <2){
            return result;
        };

        std::vector<double> prefixSumG, prefixSumH;
        prefixSumG.resize(sorted_G_vals.size());
        prefixSumH.resize(sorted_H_vals.size());
        std::inclusive_scan(std::execution::par, sorted_G_vals.begin(), sorted_G_vals.end(), prefixSumG.begin());
        std::inclusive_scan(std::execution::par, sorted_H_vals.begin(), sorted_H_vals.end(), prefixSumH.begin());
        bool all_same = false;
        double first_val = this->dataset_[sorted_feature_indices_vec.front()][feature];
        double last_val = this->dataset_[sorted_feature_indices_vec.back()][feature];
        if (first_val == last_val){
            all_same = true;
        };
        if(all_same){
            return result;
        };
        std::vector<int> index_vec(sorted_feature_indices_vec.size()-1);
        std::iota(index_vec.begin(), index_vec.end(), 0);
        auto best_pair = std::transform_reduce(
            std::execution::par, 
            index_vec.begin(),
            index_vec.end(),
            std::pair<int, double>{-1, -std::numeric_limits<double>::infinity()}, //pair is index_position and gain
            [](const std::pair<int, double>& a, const std::pair<int, double>& b)
            {
                return (a.second > b.second? a : b);
            },
            [&](int pos)
            {
                double left_val = this->dataset_[sorted_feature_indices_vec[pos]][feature];
                double right_val = this->dataset_[sorted_feature_indices_vec[pos+1]][feature]; 
                if(left_val == right_val){
                    return std::pair<int,double>(pos, -std::numeric_limits<double>::infinity());
                };

                double G_left = prefixSumG[pos];
                double H_left = prefixSumH[pos];
                double G_right = prefixSumG.back() - G_left;
                double H_right = prefixSumH.back() - H_left;
              
                  
                   
                double gain = this->computeGain(G_left, H_left, G_right, H_right);
                return std::pair<int, double>{pos, gain}; ///
            }
        );
        if (best_pair.first == -1 || best_pair.second <=0){
            return result;
        };
        int best_j = best_pair.first;
        double left_val = this->dataset_[sorted_feature_indices_vec[best_j]][feature];
        double right_val = this->dataset_[sorted_feature_indices_vec[best_j+1]][feature];
        double threshold = (left_val + right_val)/2.0;

        result.feature_index = feature;
        result.feature_threshold = threshold;
        result.gain = best_pair.second;
        return result;
    };

double Tree::computeWeight(const std::vector<int>& data){
    double sum_H = std::transform_reduce(std::execution::par, data.begin(), data.end(), 0.0, std::plus<>(), [this](int i){return this->preComputedH_[i];});
    double sum_G = std::transform_reduce(std::execution::par, data.begin(), data.end(), 0.0, std::plus<>(), [this](int i){return this->preComputedG_[i];});
    return -sum_G/(sum_H + this->lambda_);
};


double Tree::computeGain(double G_left, double H_left, double G_right, double H_right){
    double left_gain = std::pow(G_left, 2)/(H_left + this->lambda_);
    double right_gain = std::pow(G_right, 2)/(H_right + this->lambda_);
    double total_gain = std::pow(G_left+G_right, 2)/(H_left+H_right+this->lambda_);
    double gain = 0.5*(left_gain+right_gain-total_gain)-this->gamma_;
    return gain;
};

double Tree::predict(const std::vector<double>& data) const{
    if (!this->root) return 0.0;
    
    const TreeNode* node = this->root.get();
    
    while(node && !node->is_leaf){
        const int f = node->feature_index;
        const double thr = node->threshold;
        const TreeNode* next = (data[f] <= thr) ? node->left.get() : node->right.get();
        
        // If next is null but we're not at a leaf, something is wrong
        // In this case, treat current node as leaf
        if (!next) {
            break;
        }
        node = next;
    }
    
    // If we somehow ended up with nullptr, return 0
    // Otherwise return the leaf value
    return node ? node->value : 0.0;
};

void Tree::save(std::ofstream& out)const{
    const std::function<void(const TreeNode*)> dfs = [&](const TreeNode* node){
        bool exists = (node !=nullptr);
        out.write(reinterpret_cast<const char*>(&exists), sizeof(exists));
        if(!exists){
            return;
        };
        out.write(reinterpret_cast<const char*>(&node->is_leaf), sizeof(node->is_leaf));
        out.write(reinterpret_cast<const char*>(&node->feature_index), sizeof(node->feature_index));
        out.write(reinterpret_cast<const char*>(&node->threshold), sizeof(node->threshold));
        out.write(reinterpret_cast<const char*>(&node->value), sizeof(node->value));
        dfs(node->left.get());
        dfs(node->right.get());
    };
    dfs(this->root.get());
};

void Tree::load(std::ifstream& in){
    const std::function<std::unique_ptr<TreeNode>()> dfs = [&]()->std::unique_ptr<TreeNode>{
        bool exists;
        in.read(reinterpret_cast<char*>(&exists), sizeof(exists));
        if(!exists){
            return nullptr;
        }
        auto node = std::make_unique<TreeNode>();
        in.read(reinterpret_cast<char*>(&node->is_leaf), sizeof(node->is_leaf));
        in.read(reinterpret_cast<char*>(&node->feature_index), sizeof(node->feature_index));
        in.read(reinterpret_cast<char*>(&node->threshold), sizeof(node->threshold));
        in.read(reinterpret_cast<char*>(&node->value), sizeof(node->value));

        node->left = dfs();
        node->right =dfs();
        return node;
    };
    this->root = dfs();
}