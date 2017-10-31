#include <cassert>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include "neuron.hh"
#include "net.hh"

void sanity_check(){
    input_neuron input_a;
    input_neuron input_b;

    hidden_neuron<step_function> hidden_a;
    hidden_a.add_input(input_a, 2);
    hidden_a.add_input(input_b, 2);
    hidden_a.set_bias(-1);

    hidden_neuron<step_function> hidden_b;
    hidden_b.add_input(input_a, -2);
    hidden_b.add_input(input_b, -2);
    hidden_b.set_bias(3);

    hidden_neuron<step_function> output_neuron;
    output_neuron.add_input(hidden_a, 1);
    output_neuron.add_input(hidden_b, 1);
    output_neuron.set_bias(-2);

    std::vector<hidden_neuron<step_function>*> hidden_neurons{&hidden_a, &hidden_b, &output_neuron};

    input_a.set_value(0);
    input_b.set_value(0);
    for(auto neu: hidden_neurons){
        neu->update();
    }
    assert(output_neuron.output() == 0);

    input_a.set_value(0);
    input_b.set_value(1);
    for(auto neu: hidden_neurons){
        neu->update();
    }
    assert(output_neuron.output() == 1);

    input_a.set_value(1);
    input_b.set_value(0);
    for(auto neu: hidden_neurons){
        neu->update();
    }
    assert(output_neuron.output() == 1);

    input_a.set_value(1);
    input_b.set_value(1);
    for(auto neu: hidden_neurons){
        neu->update();
    }
    assert(output_neuron.output() == 0);
}

int main() {
//    sanity_check();

    net<784, 800, 10> net;

    net.set_weight<1, 0, 0>(2);
    net.set_weight<1, 0, 1>(2);
    net.set_bias<1, 0>(-1);
    net.set_weight<1, 1, 0>(-2);
    net.set_weight<1, 1, 1>(-2);
    net.set_bias<1, 1>(3);
    net.set_weight<2, 0, 0>(1);
    net.set_weight<2, 0, 1>(1);
    net.set_bias<2, 0>(-2);

    constexpr int iterations = 10000000;

    std::mt19937 random;
    std::uniform_int_distribution dist(0, 1);

    {
        auto begin = std::clock();
        for(int i = 0; i < iterations; ++i) {
            auto a = random() & 1;
            auto b = random() & 1;

            net.set_input<0>(a);
            net.set_input<1>(b);
            net.evaluate();
            assert(net.get_output<0>() == a ^ b);
        }
        auto end = std::clock();
        std::cout << "template: " << (1000. * (end - begin) / CLOCKS_PER_SEC) << std::endl;
    }

    input_neuron input_a;
    input_neuron input_b;

    hidden_neuron<step_function> hidden_a;
    hidden_a.add_input(input_a, 2);
    hidden_a.add_input(input_b, 2);
    hidden_a.set_bias(-1);

    hidden_neuron<step_function> hidden_b;
    hidden_b.add_input(input_a, -2);
    hidden_b.add_input(input_b, -2);
    hidden_b.set_bias(3);

    hidden_neuron<step_function> output_neuron;
    output_neuron.add_input(hidden_a, 1);
    output_neuron.add_input(hidden_b, 1);
    output_neuron.set_bias(-2);

    std::vector<hidden_neuron<step_function>*> hidden_neurons{&hidden_a, &hidden_b, &output_neuron};
    {
        auto begin = std::clock();

        for(int i = 0; i < iterations; ++i) {
            auto a = random() & 1;
            auto b = random() & 1;
            input_a.set_value(a);
            input_b.set_value(b);
            for(auto neu: hidden_neurons) {
                neu->update();
            }
            assert(output_neuron.output() == a ^ b);
        }

        auto end = std::clock();
        std::cout << "naive: " << (1000. * (end - begin) / CLOCKS_PER_SEC) << std::endl;
    }
    return 0;
}
