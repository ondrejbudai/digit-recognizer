#ifndef DIGIT_RECOGNIZER_NET_HH
#define DIGIT_RECOGNIZER_NET_HH

#include "meta_helpers.hh"

#include <array>
#include <tuple>
#include <functional>

struct step_function {
    static double eval(double input){
        return input >= 0 ? 1 : 0;
    }
};

struct sigmoid_function {
    static double eval(double input){
        return 1. / (1. + exp(-input));
    }

    static double derivative(double input){
        return input * (1 - input);
    }
};

struct hyperbolic_tangent {
    static double eval(double input){
        return tanh(input);
    }

    static double derivative(double input){
        return 1 - input * input;
    }
};

template<size_t neuron_count_in_previous_layer>
struct neuron {
    neuron(){
        weights.fill(0);
        deltas.fill(0);
    }
    std::array<double, neuron_count_in_previous_layer> weights{};
    std::array<double, neuron_count_in_previous_layer> deltas{};
    std::array<double, neuron_count_in_previous_layer> current_deltas{};
    double output = 0.;
    double bias = 0.;
    double bias_current_delta = 0.;
    double biasdelta = 0.;
    double gradient = 0.;
    bool enabled = true;
};

template<size_t neuron_count_, size_t neuron_count_in_previous_layer>
struct layer {
    static constexpr size_t neuron_count = neuron_count_;
    std::array<neuron<neuron_count_in_previous_layer>, neuron_count> neurons;

    template<size_t neuron_number>
    auto& get_neuron(){
        static_assert(neuron_number < neuron_count);
        return neurons[neuron_number];
    }

    template<size_t neuron_number>
    const auto& get_neuron() const{
        static_assert(neuron_number < neuron_count);
        return neurons[neuron_number];
    }

    auto& get_neuron(size_t neuron_number){
        return neurons[neuron_number];
    }
    const auto& get_neuron(size_t neuron_number) const{
        return neurons[neuron_number];
    }
};

template<typename ActivationFunction, size_t... args>
class net {
public:
    template<size_t input_number>
    void set_input(double value){
        auto& layer = get_layer<0>();
        auto& neuron = layer.template get_neuron<input_number>();
        neuron.output = value;
    }

    void set_input(size_t input_number, double value){
        auto& layer = get_layer<0>();
        auto& neuron = layer.get_neuron(input_number);
        neuron.output = value;
    }

    template<size_t output_number>
    double get_output() const {
        const auto& layer = get_layer<layer_count - 1>();
        const auto& neuron = layer.template get_neuron<output_number>();

        return neuron.output;
    }

    double get_output(size_t output_number) const {
        const auto& layer = get_layer<layer_count - 1>();
        const auto& neuron = layer.get_neuron(output_number);

        return neuron.output;
    }

    template<size_t output_number>
    void set_target_output(double value){
        static_assert(output_number < output_count);
        target_values[output_number] = value;
    }

    void set_target_output(size_t output_number, double value){
        target_values[output_number] = value;
    }

    template<size_t layer_number, size_t neuron_number, size_t input_number>
    void set_weight(double weight){
        auto& layer = get_layer<layer_number>();
        auto& neuron = layer.template get_neuron<neuron_number>();
        neuron.weights[input_number] = weight;
    };

    template<size_t layer_number, size_t neuron_number, size_t input_number>
    double get_weight(){
        auto& layer = get_layer<layer_number>();
        auto& neuron = layer.template get_neuron<neuron_number>();
        return neuron.weights[input_number];
    };

    template<size_t layer_number, size_t neuron_number>
    void set_bias(double bias){
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

    void update_weights() {
        update_weights_layers();
        current_batch = (current_batch + 1) % batch_size;
    }

    void random_initialize(const std::function<double()>& random_gen){
        random_initialize_layers(random_gen);
    }

    constexpr auto get_output_count() const{
        return output_count;
    }

    void set_learning_rate(double new_learning_rate){
        learning_rate = new_learning_rate;
    }

    double get_learning_rate(){
        return learning_rate;
    }

    bool dropout_enabled = false;

private:
    static constexpr size_t layer_count = sizeof...(args);
    static_assert(layer_count >= 3);

    typename zip<size_t, std::tuple, layer, args...>::template with<typename remove_last_from_size_tuple<args...>::type>::type layers;

    static constexpr size_t output_count = size_tuple_element<layer_count - 1, size_tuple<args...>>::value;
    std::array<double, output_count> target_values{};

    double learning_rate = 0;
    double dropout_probability = 0.1;

    std::mt19937 rd{std::random_device{}()};
    std::uniform_real_distribution<double> dis{0, 1};
    size_t batch_size = 1;
    size_t current_batch = 0;

    template<size_t layer_number>
    static constexpr size_t get_neuron_count_at_layer(){
        return std::tuple_element<layer_number, decltype(layers)>::type::neuron_count;
    }

    template<size_t layer_number>
    auto& get_layer(){
        return std::get<layer_number>(layers);
    }

    template<size_t layer_number>
    const auto& get_layer() const {
        return std::get<layer_number>(layers);
    }

    template<size_t layer_number = 1>
    void evaluate_layers(){
        static_assert(layer_number > 0, "First layer cannot be evaluated!");

        for(size_t neuron = 0; neuron < get_neuron_count_at_layer<layer_number>(); ++neuron){
            evaluate_neuron<layer_number>(neuron);
        }

        if constexpr(layer_number < layer_count - 1){
            evaluate_layers<layer_number+1>();
        }
    }

    template<size_t layer_number>
    void evaluate_neuron(size_t neuron_number){
        auto& neuron = get_layer<layer_number>().neurons[neuron_number];
        double value = neuron.bias;
        if(dropout_enabled) {
            neuron.enabled = dis(rd) < (1 - dropout_probability);
        } else {
            neuron.enabled = true;
        }
        if(!neuron.enabled){
            neuron.output = 0;
            return;
        }

        for(size_t input_number = 0; input_number < get_neuron_count_at_layer<layer_number-1>(); ++input_number){
            value += get_layer<layer_number - 1>().neurons[input_number].output * neuron.weights[input_number];
        }

        if(dropout_enabled){
            value /= (1 - dropout_probability);
        }

        neuron.output = ActivationFunction::eval(value);
    }

    void calc_gradient_output_layer(){
        //TODO: move me to fields
        auto& output_layer = get_layer<layer_count - 1>();

        for(size_t output_neuron = 0; output_neuron < output_count; ++output_neuron){
            auto& neuron = output_layer.get_neuron(output_neuron);
            auto error = target_values[output_neuron] - neuron.output;
            double gradient = error /** std::abs(error)*/ * ActivationFunction::derivative(neuron.output);
            neuron.gradient = gradient;
        }
    }

    template<size_t layer_number = layer_count - 2>
    void calc_gradient_hidden_layers(){
        for(size_t neuron_number = 0; neuron_number < get_neuron_count_at_layer<layer_number>(); ++neuron_number){
            calc_gradient_hidden_layer_neuron<layer_number>(neuron_number);
        }

        if constexpr (layer_number > 1){
            calc_gradient_hidden_layers<layer_number - 1>();
        }
    }

    template<size_t layer_number>
    void calc_gradient_hidden_layer_neuron(size_t neuron_number){
        auto& layer = get_layer<layer_number>();
        auto& next_layer = get_layer<layer_number + 1>();

        auto& neuron = layer.get_neuron(neuron_number);
        if(!neuron.enabled){
            return;
        }
        double gradient = 0;
        for(size_t output_neuron_number = 0; output_neuron_number < get_neuron_count_at_layer<layer_number + 1>(); ++output_neuron_number){
            const auto& output_neuron = next_layer.get_neuron(output_neuron_number);
            gradient += output_neuron.gradient * output_neuron.weights[neuron_number];
        }

        gradient *= ActivationFunction::derivative(neuron.output);

        neuron.gradient = gradient;
    }

    template<size_t layer_number = 1>
    void update_weights_layers(){
        for(size_t neuron_number = 0; neuron_number < get_neuron_count_at_layer<layer_number>(); ++neuron_number){
            update_weights_layer_neuron<layer_number>(neuron_number);
        }

        if constexpr (layer_number < layer_count - 1){
            update_weights_layers<layer_number + 1>();
        }
    }

    template<size_t layer_number>
    void update_weights_layer_neuron(const size_t neuron_number){
        auto& layer = get_layer<layer_number>();
        auto& prev_layer = get_layer<layer_number-1>();
        auto& neuron = layer.get_neuron(neuron_number);

        if(!neuron.enabled){
            return;
        }

        for(size_t input_neuron = 0; input_neuron < get_neuron_count_at_layer<layer_number - 1>(); ++input_neuron){
            if(!prev_layer.neurons[input_neuron].enabled){
                continue;
            }
            double newdelta = learning_rate * prev_layer.neurons[input_neuron].output * neuron.gradient;
            neuron.current_deltas[input_neuron] += newdelta;

            if(current_batch == batch_size - 1){
                neuron.current_deltas[input_neuron] += 0.8 * neuron.deltas[input_neuron];
                neuron.weights[input_neuron] += neuron.current_deltas[input_neuron]/* - neuron.weights[input_neuron] * 0.0000005*/;
                neuron.deltas[input_neuron] = neuron.current_deltas[input_neuron];
                neuron.current_deltas[input_neuron] = 0;
            }
        }
        neuron.bias_current_delta += learning_rate * neuron.bias * neuron.gradient;
        if(current_batch == batch_size - 1){
            neuron.bias_current_delta += 0.8 * neuron.biasdelta;
            neuron.bias += neuron.bias_current_delta/* - neuron.bias * 0.000001*/;
            neuron.biasdelta = neuron.bias_current_delta;
            neuron.bias_current_delta = 0;
        }
    }

    template <size_t layer_number = 1>
    void random_initialize_layers(const std::function<double()>& random_gen){
        for(size_t neuron_number = 0; neuron_number < get_neuron_count_at_layer<layer_number>(); ++neuron_number){
            auto& layer = get_layer<layer_number>();
            auto& neuron = layer.get_neuron(neuron_number);

            std::generate(std::begin(neuron.weights), std::end(neuron.weights), random_gen);

            neuron.bias = random_gen();
        }

        if constexpr (layer_number < layer_count - 1){
            random_initialize_layers<layer_number + 1>(random_gen);
        }
    }
};

#endif //DIGIT_RECOGNIZER_NET_HH
