#include <cassert>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
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

int main() {
    sanity_check();

    //mnist: 784-800-10
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

    net<sigmoid_function, 2, 4, 1> net2;
    net2.calc_gradient();
    return 0;
}
