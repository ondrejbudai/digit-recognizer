#include <cassert>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <iomanip>
#include "net.hh"

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
    std::uniform_int_distribution dist(0, 1);

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

int main() {

    std::cout << std::setprecision(20);
    //mnist: 784-800-10

    sanity_check();

    net<hyperbolic_tangent, 2, 4, 1> xor_net;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);

    xor_net.random_initialize([&](){
        return dis(gen);
    });

    constexpr size_t iterations = 10000;
    std::uniform_int_distribution dis_int(0, 1);

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



    return 0;
}
