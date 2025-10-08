#include "model.h"
#include <fstream>
XGBoost::XGBoost() = default;
XGBoost::XGBoost(int n_estimator, int max_depth, double learning_rate, double gamma, double lambda ) : \
n_estimators_(n_estimator), max_depth_(max_depth), learning_rate_(learning_rate), gamma_(gamma), lambda_(lambda){

};

void XGBoost::train(std::vector<std::vector<double>>& train_data, const std::vector<double>&train_label){
    std::println("Starting training...");
    // For my code, I will use mean value as initial y hat
    double y_mean = (std::accumulate(train_label.begin(), train_label.end(), 0.0))/(double)(train_label.size());
    this->base_pred_ = y_mean;
    std::vector<double> y_hat (train_label.size(), y_mean);
    std::vector<double> preComputedG(train_label.size());
    std::vector<double> preComputedH(train_label.size(), 1.0);
    std::vector<int> indices_vec(train_data.size());
    std::iota(indices_vec.begin(), indices_vec.end(), 0);
    std::for_each(std::execution::par,      
        indices_vec.begin(),
        indices_vec.end(), 
        [&](int idx){
            preComputedG[idx] = y_hat[idx] - train_label[idx];
        });
    std::println("Initial gradient stats:");
    std::println("  Min G: {}", *std::min_element(preComputedG.begin(), preComputedG.end()));
    std::println("  Max G: {}", *std::max_element(preComputedG.begin(), preComputedG.end()));
    std::println("  Mean G: {}", std::accumulate(preComputedG.begin(), preComputedG.end(), 0.0) / (double)preComputedG.size());
    std::println("  base_pred: {}", this->base_pred_);
    std::vector<int> dataset_indices(train_data.size());
    std::iota(dataset_indices.begin(), dataset_indices.end(), 0);
    std::vector<std::vector<int>> sorted_dataset_indices(train_data[0].size(), dataset_indices);
    std::vector<int> feat_vec(train_data[0].size(), 0);
    std::iota(feat_vec.begin(), feat_vec.end(), 0);
    std::for_each(std::execution::par, feat_vec.begin(), feat_vec.end(), [&](int feat_idx){
        std::sort(sorted_dataset_indices[feat_idx].begin(), sorted_dataset_indices[feat_idx].end(), [&](int a, int b){
            const double va = train_data[a][feat_idx];
            const double vb = train_data[b][feat_idx];
            if (va< vb) return true;
            if (vb< va) return false;
            return a < b;
        });
    });

    for (int i = 0; i< this->n_estimators_; i++){
        std::println("Building tree {}", i+1);
        std::unique_ptr<Tree> tree = std::make_unique<Tree>(this->max_depth_, this->gamma_, this->lambda_, train_data, sorted_dataset_indices, preComputedG, preComputedH, this->thread_pool);
        tree->build(indices_vec);
        std::for_each(std::execution::par, indices_vec.begin(), indices_vec.end(), [&](int index){
            y_hat[index]+= this->learning_rate_*(tree->predict(train_data[index]));
        });
        std::for_each(std::execution::par, \
            indices_vec.begin(), \
            indices_vec.end(), \
            [&](int idx){
                preComputedG[idx] = y_hat[idx] - train_label[idx];
            }
        );
        this->trees.push_back(std::move(tree));
    };
    std::println("Train completed");
};
double XGBoost::predict(const std::vector<double>& data)const{
    return (this->base_pred_ + std::transform_reduce(std::execution::par, this->trees.begin(), this->trees.end(), 0.0, std::plus<>(), [&](const std::unique_ptr<Tree>& tree){
        return this->learning_rate_*(tree->predict(data));
    }));
}
void XGBoost::test(const std::vector<std::vector<double>>& test_data,
                   const std::vector<double>& test_label) {
    std::vector<double> preds(test_label.size(), 0.0);
    std::vector<int> idx_vec(test_label.size());
    std::iota(idx_vec.begin(), idx_vec.end(), 0);

    std::for_each(std::execution::par, idx_vec.begin(), idx_vec.end(), [&](int idx) {
        preds[idx] = this->predict(test_data[idx]);
    });




    // Compute SS_res = sum((y_true - y_pred)^2)
    double ss_res = std::transform_reduce(
        std::execution::par,
        idx_vec.begin(), idx_vec.end(),
        0.0,
        std::plus<>(),
        [&](int i) {
            double diff = test_label[i] - preds[i];
            return diff * diff;
        });

    // Compute SS_tot = sum((y_true - mean_y)^2)
    double ss_tot = std::transform_reduce(
        std::execution::par,
        test_label.begin(), test_label.end(),
        0.0,
        std::plus<>(),
        [&](double y) {
            double diff = y - this->base_pred_;
            return diff * diff;
        });

    double r2 = 1.0 - (ss_res / ss_tot);

    std::println("R² score: {:.6f}", r2);

    // (Optional) Show few predictions
    for (size_t i = 0; i < std::min<size_t>(5, test_label.size()); ++i) {
        std::println("Pred: {:>10.4f} | Actual: {:>10.4f}", preds[i], test_label[i]);
    }
};

void XGBoost::saveModel(const std::filesystem::path& filename)const{
    std::ofstream out (filename, std::ios::binary);
    if(!out.is_open()){
        std::cerr << "Failed to open path " << filename << " for writing\n";
        return;
    };
    out.write(reinterpret_cast<const char*>(&this->base_pred_), sizeof(this->base_pred_));
    out.write(reinterpret_cast< const char*>(&this->n_estimators_), sizeof(this->n_estimators_));
    out.write(reinterpret_cast<const char*>(&this->max_depth_), sizeof(this->max_depth_));
    out.write(reinterpret_cast<const char*>(&this->learning_rate_), sizeof(this->learning_rate_));
    out.write(reinterpret_cast<const char*>(&this->gamma_), sizeof(this->gamma_));
    out.write(reinterpret_cast<const char*>(&this->lambda_), sizeof(this->lambda_));

    size_t treeCounts = this->trees.size();
    out.write(reinterpret_cast<const char*>(&treeCounts), sizeof(treeCounts));
    for(const auto& tree: this->trees){
        tree->save(out);
    };

    out.close();
    std::println("Model saved in binary format to {}\n", filename.string());
}

void XGBoost::loadModel(const std::filesystem::path& filename){
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){
        std::cerr << "Failed to open " << filename << " for reading\n";
        return;
    }
    in.read(reinterpret_cast<char*>(&this->base_pred_), sizeof(base_pred_));
    in.read(reinterpret_cast<char*>(&this->n_estimators_), sizeof(this->n_estimators_));
    in.read(reinterpret_cast<char*>(&this->max_depth_), sizeof(this->max_depth_));
    in.read(reinterpret_cast<char*>(&this->learning_rate_), sizeof(this->learning_rate_));
    in.read(reinterpret_cast<char*>(&this->gamma_), sizeof(this->gamma_));
    in.read(reinterpret_cast<char*>(&this->lambda_), sizeof(this->lambda_));
    size_t treeCounts;
    in.read(reinterpret_cast<char*>(&treeCounts), sizeof(treeCounts));
    this->trees.clear();
    this->trees.reserve(treeCounts);

    for(size_t i = 0; i < treeCounts; ++i){
        auto tree = std::make_unique<Tree>(
            this->max_depth_,
            this->gamma_,
            this->lambda_,
            this->dummy_dataset_,
            this->dummy_sorted_indices_, 
            this->dummy_G_,
            this->dummy_H_,
            this->thread_pool 
        );
        tree->load(in);
        this->trees.push_back(std::move(tree));
    }

    in.close();
    std::println("Successfully loaded model from path {}", filename.string());


}