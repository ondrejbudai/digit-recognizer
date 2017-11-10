#include <cassert>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <optional>
#include "net.hh"

constexpr size_t input_size = 784;
struct sample_t {
    std::array<double, input_size> inputs;
    size_t target = 0;
};

class dataset_t {
public:


    void load(std::string_view filename_vectors, std::string_view filename_labels){
        load_vectors(filename_vectors);
        load_labels(filename_labels);
    }
    std::vector<sample_t> data;
private:

    void load_vectors(std::string_view filename_vectors) {
        std::ifstream input_stream{std::string{filename_vectors}};
        if(!input_stream.is_open()){
            std::cerr << "Cannot open dataset: " << filename_vectors << std::endl;
            throw;
        }

        int tmp;
        data.emplace_back();
        size_t col = 0;
        while(input_stream >> tmp){
            data.back().inputs[col] = tmp / 255.0;
            col++;

            int next_char = input_stream.get();
            if(next_char == '\n'){
                if(col != data.back().inputs.size() - 1){
                    throw;
                }
                col = 0;
                data.emplace_back();
            } else if(next_char == '\r'){
                if(input_stream.get() != '\n'){
                    throw;
                }
                col = 0;
                data.emplace_back();
            } else if(next_char != ','){
                throw;
            }
        }

        if(!input_stream.eof()){
            std::cerr << "Reading dataset failed: " << filename_vectors << std::endl;
            throw;
        }
    }

    void load_labels(std::string_view filename_labels){
        std::ifstream input_stream{std::string{filename_labels}};

        unsigned tmp;
        size_t row = 0;
        while(input_stream >> tmp){
            data[row].target = tmp;
            row++;
            if(row >= std::size(data)){
                throw;
            }
        }

        if(row != std::size(data) - 1){
            throw;
        }

        if(!input_stream.eof()){
            throw;
        }
    }
};

void sanity_check(){
    net<step_function, 2, 2, 1> net;

    net.set_weight<1, 0, 0>(2);
    net.set_weight<1, 0, 1>(2);
    net.set_bias<1, 0>(-1);
    net.set_weight<1, 1, 0>(-2);
    net.set_weight<1, 1, 1>(-2);
    net.set_bias<1, 1>(3);
    net.set_weight<2, 0, 0>(1);
    net.set_weight<2, 0, 1>(1);
    net.set_bias<2, 0>(-2);

    net.set_input<0>(0);
    net.set_input<1>(0);
    net.evaluate();
    assert(net.get_output<0>() == 0);

    net.set_input<0>(0);
    net.set_input<1>(1);
    net.evaluate();
    assert(net.get_output<0>() == 1);

    net.set_input<0>(1);
    net.set_input<1>(0);
    net.evaluate();
    assert(net.get_output<0>() == 1);

    net.set_input<0>(1);
    net.set_input<1>(1);
    net.evaluate();
    assert(net.get_output<0>() == 0);
}

void benchmark(){
    net<step_function, 2, 2, 1> network;

    network.set_weight<1, 0, 0>(2);
    network.set_weight<1, 0, 1>(2);
    network.set_bias<1, 0>(-1);
    network.set_weight<1, 1, 0>(-2);
    network.set_weight<1, 1, 1>(-2);
    network.set_bias<1, 1>(3);
    network.set_weight<2, 0, 0>(1);
    network.set_weight<2, 0, 1>(1);
    network.set_bias<2, 0>(-2);

    constexpr int iterations = 10000000;

    std::mt19937 random;
    std::uniform_int_distribution<int> dist(0, 1);

    {
        auto begin = std::clock();
        for(int i = 0; i < iterations; ++i) {
            auto a = random() & 1;
            auto b = random() & 1;

            network.set_input<0>(a);
            network.set_input<1>(b);
            network.evaluate();
            assert(network.get_output<0>() == (a ^ b));
        }
        auto end = std::clock();
        std::cout << "template: " << (1000. * (end - begin) / CLOCKS_PER_SEC) << std::endl;
    }
}

void xor_test() {
    net<hyperbolic_tangent, 2, 4, 1> xor_net;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);

    xor_net.random_initialize([&](){
        return dis(gen);
    });

    constexpr size_t iterations = 10000;
    std::uniform_int_distribution<int> dis_int(0, 1);

    for(size_t i = 0; i < iterations; ++i){
        int a = dis_int(gen);
        int b = dis_int(gen);

        xor_net.set_input<0>(a);
        xor_net.set_input<1>(b);
        xor_net.evaluate();
        xor_net.set_target_output<0>(a ^ b);
        xor_net.calc_gradient();
        xor_net.update_weights();
    }

    int hits = 0;
    int undecided = 0;

    for(size_t i = 0; i < iterations; ++i){
        int a = dis_int(gen);
        int b = dis_int(gen);

        xor_net.set_input<0>(a);
        xor_net.set_input<1>(b);
        xor_net.evaluate();
        double raw_output = xor_net.get_output<0>();
        if(raw_output > 0.1 && raw_output < 0.9){
            undecided++;
            continue;
        }
        int output = raw_output <= 0.1 ? 0 : 1;
        hits += output == (a ^ b);
    }

    std::cout << "Correct guesses: " << hits << std::endl;
    std::cout << "Undecided:       " << undecided << std::endl;
}

template<typename Net>
size_t get_best_output(const Net& net){
    auto max = std::numeric_limits<double>::lowest();
    auto best_output = size_t{0};
    for(size_t output_number = 0; output_number < net.get_output_count(); ++output_number){
        auto output = net.get_output(output_number);
        if(output > max){
            max = output;
            best_output = output_number;
        }
    }

    return best_output;
}

int main() {

    std::cout << std::setprecision(20);
    //mnist: 784-800-10

//    sanity_check();

    dataset_t dataset{};
    dataset.load("MNIST_DATA/mnist_train_vectors.csv", "MNIST_DATA/mnist_train_labels.csv");

    auto network = std::make_unique<net<hyperbolic_tangent, 784, 400, 200, 10>>();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0, 0.05);

    network->random_initialize([&](){
        return dis(gen);
    });

    network->set_learning_rate(0.001);

    size_t last_target = 0;
    int iterations = 0;
    for(const auto& sample: dataset.data) {
        size_t input_number = 0;
        for(auto input: sample.inputs) {
            network->set_input(input_number++, input);
        }
        network->evaluate();

        network->set_target_output(last_target, 0);
        last_target = static_cast<size_t>(sample.target);
        network->set_target_output(last_target, 1);

        network->calc_gradient();
        network->update_weights();

        if(iterations % 6000 == 0) {
//            network->set_learning_rate(network->get_learning_rate() * 0.5);
            std::cerr << "Learning: " << iterations / 600 << "%" << std::endl;
        }
        ++iterations;
    }

    auto test_dataset = dataset_t{};
    int ok = 0;
    test_dataset.load("MNIST_DATA/mnist_test_vectors.csv", "MNIST_DATA/mnist_test_labels.csv");

    network->dropout_enabled = false;
    for(const auto& sample: test_dataset.data){
        size_t input_number = 0;
        for(auto input: sample.inputs){
            network->set_input(input_number++, input);
        }

        network->evaluate();
        auto result = get_best_output(*network);
        if(sample.target == result){
            ++ok;
        }
    }

    std::cout << "Total: " << std::size(test_dataset.data) << " , ok: " << ok << std::endl;


    return 0;
}
