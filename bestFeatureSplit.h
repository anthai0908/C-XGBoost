#ifndef FEATURE_SPLIT
#define FEATURE_SPLIT
struct bestFeatureSplit {
    int feature_index;
    double feature_threshold;
    double gain;
};
#endif