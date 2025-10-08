#ifndef THREAD_POOL
#define THREAD_POOL
#include <thread>
#include <vector>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <type_traits>
#include <print>
class ThreadPool{
public: 
    ThreadPool(size_t thread_nums) : thread_nums_(thread_nums){
        std::println("Initialsing threadpool...");
        for (size_t i = 0 ; i <this->thread_nums_; i++){
            workers_.emplace_back([this](){
                while (true){
                     std::function<void()> task;
                    {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this](){
                        return this->stop_ || !tasks.empty();
                    });
                    if(this->stop_ && tasks.empty()){
                        return;
                    }
                    task = std::move(tasks.front());
                    tasks.pop();
                    lock.unlock();
                    }
                    task();
                }
            });
        };
        std::println("Finishing threadpool initialisation");
    };
    template<typename F, typename... Args>
    std::future<std::invoke_result_t<F, Args...>> enqueue(F&& f, Args&&... args){
        using return_type = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<return_type()>>([f = std::forward<F>(f), ...args = std::forward<Args>(args)](){
            return std::invoke(std::move(f), std::move(args)...);
        });
        std::future<return_type> ret = task->get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if(stop_){
                throw std::runtime_error("enqueue on stopped ThreadPool");
            };
            tasks.emplace([task](){
                (*task)();
            });
        }
        condition_.notify_one();
        return ret;
    }
    ~ThreadPool(){
        {  
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
         }
            
        condition_.notify_all();
    }
private:
    size_t thread_nums_;
    bool stop_ =false;
    std::queue<std::function<void()>> tasks;
    std::vector<std::jthread> workers_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;


};
#endif 