#ifndef DIGIT_RECOGNIZER_NET_HH
#define DIGIT_RECOGNIZER_NET_HH

#include "meta_helpers.hh"

#include <array>
#include <tuple>

struct step_function {
    static double eval(double input){
        return 1 ? input >= 0 : 0;
    }

    static double derivative(double){
        std::abort();
    }
};

struct sigmoid_function {
    static double eval(double input){
        return 1 / (1 + exp(-input));
    }

    static double derivative(double input){
        return input * (1 - input);
    }
};

template<size_t neuron_count_in_previous_layer> class neuron {
public:
    neuron(){
        weights.fill(0);
    }
    std::array<double, neuron_count_in_previous_layer> weights;
    double output;
    double bias;
    double gradient;
};

template<size_t neuron_count_, size_t neuron_count_in_previous_layer>
class layer {
public:
    static constexpr size_t neuron_count = neuron_count_;
    std::array<neuron<neuron_count_in_previous_layer>, neuron_count> neurons;
    template<size_t neuron_number> auto& get_neuron(){
        static_assert(neuron_number < neuron_count);
        return neurons[neuron_number];
    }

    auto& get_neuron(size_t neuron_number){
        return neurons[neuron_number];
    }
};

template<typename ActivationFunction, size_t... args> class net {
public:
    template<size_t input_number> void set_input(double value){
        auto& layer = get_layer<0>();
        auto& neuron = layer.template get_neuron<input_number>();
        neuron.output = value;
    }

    template<size_t output_number> double get_output(){
        auto& layer = get_layer<layer_count - 1>();
        auto& neuron = layer.template get_neuron<output_number>();

        return neuron.output;
    }

    template<size_t output_number> void set_target_output(double value){
        target_values[output_number] = value;
    }

    template<size_t layer_number, size_t neuron_number, size_t input_number> void set_weight(double weight){
        auto& layer = get_layer<layer_number>();
        auto& neuron = layer.template get_neuron<neuron_number>();
        neuron.weights[input_number] = weight;
    };

    template<size_t layer_number, size_t neuron_number> void set_bias(double bias){
        auto& layer = get_layer<layer_number>();
        auto& neuron = layer.template get_neuron<neuron_number>();
        neuron.bias = bias;
    };

    void evaluate() {
        evaluate_layers();
    }

    void calc_gradient() {
        calc_gradient_output_layer();
        calc_gradient_hidden_layers();
    }

private:
    static constexpr size_t layer_count = sizeof...(args);
    static_assert(layer_count >= 3);

    typename zip<size_t, std::tuple, layer, args...>::template with<typename remove_last_from_size_tuple<args...>::type>::type layers;

    static constexpr size_t output_count = size_tuple_element<layer_count - 1, size_tuple<args...>>::value;
    std::array<double, output_count> target_values;

    template<size_t layer_number> static constexpr size_t get_neuron_count_at_layer(){
        return std::tuple_element<layer_number, decltype(layers)>::type::neuron_count;
    }

    template<size_t layer_number> auto& get_layer(){
        return std::get<layer_number>(layers);
    }

    template<size_t layer_number = 1> void evaluate_layers(){
        static_assert(layer_number > 0, "First layer cannot be evaluated!");

        for(size_t neuron = 0; neuron < get_neuron_count_at_layer<layer_number>(); ++neuron){
            evaluate_neuron<layer_number>(neuron);
        }

        if constexpr(layer_number < layer_count - 1){
            evaluate_layers<layer_number+1>();
        }
    }

    template<size_t layer_number> void evaluate_neuron(size_t neuron_number){
        double value = get_layer<layer_number>().neurons[neuron_number].bias;

        for(size_t input_number = 0; input_number < get_neuron_count_at_layer<layer_number-1>(); ++input_number){
            value += get_layer<layer_number - 1>().neurons[input_number].output * get_layer<layer_number>().neurons[neuron_number].weights[input_number];
        }

        get_layer<layer_number>().neurons[neuron_number].output = ActivationFunction::eval(value);
    }

    void calc_gradient_output_layer(){
        //TODO: move me to fields
        auto& output_layer = get_layer<output_count - 1>();

        for(size_t output_neuron = 0; output_neuron < output_count; ++output_neuron){
            auto& neuron = output_layer.get_neuron(output_neuron);
            double gradient = (target_values[output_neuron] - neuron.output) * ActivationFunction::derivative(neuron.output);
            neuron.gradient = gradient;
        }
    }

    template<size_t layer_number = layer_count - 2> void calc_gradient_hidden_layers(){
        auto& layer = get_layer<layer_number>();
        auto& next_layer = get_layer<layer_number + 1>();
        for(size_t neuron_number = 0; neuron_number < get_neuron_count_at_layer<layer_number>(); ++neuron_number){
            auto& neuron = layer.get_neuron(neuron_number);
            double gradient = 0;
            for(size_t output_neuron_number = 0; output_neuron_number < get_neuron_count_at_layer<layer_number + 1>(); ++output_neuron_number){
                const auto& output_neuron = next_layer.get_neuron(output_neuron_number);
                gradient += output_neuron.gradient + output_neuron.weights[neuron_number];
            }

            neuron.gradient = gradient * ActivationFunction::derivative(neuron.output);
        }

        if constexpr (layer_number > 1){
            calc_gradient_hidden_layers<layer_number - 1>();
        }
    }
};

#endif //DIGIT_RECOGNIZER_NET_HH
