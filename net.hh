/**
 * MNIST dataset classifier
 * @author: Ondrej Budai <budai@mail.muni.cz
 */

#ifndef DIGIT_RECOGNIZER_NET_HH
#define DIGIT_RECOGNIZER_NET_HH

#include "meta_helpers.hh"
#include "array_operations.hh"

#include <array>
#include <tuple>
#include <functional>

struct step_function {
    template<size_t Values, typename Type>
    static void eval(vector_t<Values, Type>& input){
        input.apply([](auto input) {return input >= 0 ? decltype(input)(1.) : 0.;});
    }
};

struct sigmoid_function {
    static double eval(double input){
        return 1. / (1. + std::exp(-input));
    }

    static double derivative(double input){
        return input * (1 - input);
    }
};

struct relu {
    template<size_t Values, typename Type>
    static void eval(vector_t<Values, Type>& input){
        input.apply([](auto value){return std::max(value, decltype(value)(0.));});
    }

    template<size_t Values, typename Type>
    static void derivative(vector_t<Values, Type>& output, const vector_t<Values, Type>& input, const vector_t<Values, Type>& above){
        transform_many([](auto& output, auto& input, auto& above){
            output = above * (input > 0 ? 1. : 0.);
        }, output, input, above);
    }
};

struct hyperbolic_tangent {
    template<size_t Values, typename Type>
    static void eval(vector_t<Values, Type>& input){
        input.apply(static_cast<Type(*)(Type)>(std::tanh));
    }

    template<size_t Values, typename Type>
    static void derivative(vector_t<Values, Type>& output, vector_t<Values, Type>& input, vector_t<Values, Type>& above){
        transform_many([](auto& output, auto& input, auto& above){
            output = (1 - input * input) * above;
        }, output, input, above);
    }
};

struct softmax {
    template<size_t Values, typename Type>
    static void eval(vector_t<Values, Type>& input){
        auto maximum = input.max();

        Type sum = 0;

        input.apply([maximum, &sum](auto value) {
            auto result = std::exp(value - maximum);
            sum += result;
            return result;
        });

        input /= sum;
    }

    template<size_t Values, typename Type>
    static void derivative(vector_t<Values, Type>& output, vector_t<Values, Type>& input, vector_t<Values, Type>& above){
        // gradient, prediction, error
        auto delta = input;
        delta *= above;
        auto sum = delta.sum();
        transform_many([sum](auto& output, auto& input, auto& delta){
            output = delta - input * sum;
        }, output, input, delta);
    }
};

template<size_t neuron_count_, size_t neuron_count_in_previous_layer>
struct layer {
    static constexpr size_t neuron_count = neuron_count_;
    vector_t<neuron_count, double> outputs;
    vector_t<neuron_count, double> biases;
    vector_t<neuron_count, double> loss_biases;
    vector_t<neuron_count, double> loss_biases_m;
    vector_t<neuron_count, double> loss_biases_v;
    vector_t<neuron_count, double> biases_deltas;
    vector_t<neuron_count, double> gradients;
    matrix_t<neuron_count_in_previous_layer, neuron_count, double> weights;
    matrix_t<neuron_count_in_previous_layer, neuron_count, double> loss;
    matrix_t<neuron_count_in_previous_layer, neuron_count, double> loss_m;
    matrix_t<neuron_count_in_previous_layer, neuron_count, double> loss_v;
    matrix_t<neuron_count_in_previous_layer, neuron_count, double> weights_deltas;
};

template<typename ActivationFunction, size_t... args>
class net {
public:
    void set_input(size_t input_number, double value){
        auto& layer = get_layer<0>();
        layer.outputs[input_number] = value;
    }

    double get_output(size_t output_number) const {
        const auto& layer = get_layer<layer_count - 1>();
        return layer.outputs[output_number];
    }

    void set_target_output(size_t output_number, double value){
        target_values[output_number] = value;
    }

    template<size_t layer_number>
    void set_weight(size_t neuron_number, size_t connection_number, double new_weight){
        auto& layer = get_layer<layer_number>();
        layer.weights.at(connection_number, neuron_number) = new_weight;
    }

    template<size_t layer_number>
    void set_bias(size_t neuron_number, double new_bias){
        auto& layer = get_layer<layer_number>();
        layer.biases[neuron_number] = new_bias;
    }


    void evaluate() {
        evaluate_layers();
    }

    void calc_gradient() {
        calc_gradient_output_layer();
        calc_gradient_hidden_layers();
    }

    void update_weights() {
        update_weights_layers();

        decay1_t *= decay1;
        decay2_t *= decay2;
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
    vector_t<output_count, double> target_values;

    double learning_rate = 0;
    double dropout_probability = 0.1;
    double t = 1;
    static constexpr auto decay1 = 0.9;
    static constexpr auto decay2 = 0.999;
    double decay1_t = decay1;
    double decay2_t = decay2;

    std::mt19937 rd{std::random_device{}()};
    std::uniform_real_distribution<double> dis{0, 1};

    template<size_t layer_number>
    static constexpr size_t get_neuron_count_at_layer(){
        return std::tuple_element<layer_number, decltype(layers)>::type::neuron_count;
    }

    template<size_t layer_number>
    constexpr auto& get_layer(){
        return std::get<layer_number>(layers);
    }

    template<size_t layer_number>
    const auto& get_layer() const {
        return std::get<layer_number>(layers);
    }

    template<size_t layer_number = 1>
    void evaluate_layers(){
        static_assert(layer_number > 0, "First layer cannot be evaluated!");

//        std::cout << "evaluatel" << layer_number << std::endl;

        auto& layer = get_layer<layer_number>();
        auto& prev_layer = get_layer<layer_number - 1>();

        layer.outputs = matrix_multiply(transpose(layer.weights), prev_layer.outputs);
        layer.outputs += layer.biases;
        if constexpr (layer_number < layer_count - 1) {
            ActivationFunction::eval(layer.outputs);
        } else {
            softmax::eval(layer.outputs);
        }

        if constexpr(layer_number < layer_count - 1){
            evaluate_layers<layer_number+1>();
        }
    }

    void calc_gradient_output_layer(){
        auto& output_layer = get_layer<layer_count - 1>();

        transform_many([](auto& target, auto& prediction){
            auto denominator = std::max(prediction - prediction * prediction, 1e-11);
            target = (prediction - target) / denominator;
        }, target_values, output_layer.outputs);

        softmax::derivative(output_layer.gradients, output_layer.outputs, target_values);
    }

    template<size_t layer_number = layer_count - 2>
    void calc_gradient_hidden_layers(){
        auto& layer = get_layer<layer_number>();
        auto& next_layer = get_layer<layer_number + 1>();

        auto a = matrix_multiply(next_layer.weights, next_layer.gradients);

        ActivationFunction::derivative(layer.gradients, layer.outputs, a);


        if constexpr (layer_number > 1){
            calc_gradient_hidden_layers<layer_number - 1>();
        }
    }

    template<size_t layer_number = 1>
    void update_weights_layers(){
        auto& layer = get_layer<layer_number>();
        auto& prev_layer = get_layer<layer_number-1>();

//        constexpr auto decay = 0.9;
        constexpr auto eps = 1e-8;

        auto gradient_matrix = matrix_multiply(prev_layer.outputs, transpose(layer.gradients));

        auto decay1_ta = 1 - decay1_t;
        auto decay2_ta = 1 - decay2_t;

        auto adam = [decay1_ta, decay2_ta](auto& gradient, auto& loss_m, auto& loss_v, auto&weight){
            loss_m = loss_m * decay1 + (1 - decay1) * gradient;
            loss_v = loss_v * decay2 + (1 - decay2) * gradient * gradient;

            auto loss_m_corrected = loss_m / (decay1_ta);
            auto loss_v_corrected = loss_v / (decay2_ta);

            weight -= 0.0001 * loss_m_corrected / (std::sqrt(loss_v_corrected) + eps);

        };

//        auto rmsprop = [](auto& gradient, auto& loss, auto& weight){
//            loss = decay * loss + (1 - decay) * gradient * gradient;
//            weight -= 0.0001 * gradient / (std::sqrt(loss) + eps);
//        };


        transform_many(adam, gradient_matrix, layer.loss_m, layer.loss_v, layer.weights);
//        transformer3{gradient_matrix, layer.loss, layer.weights}.apply(rmsprop);

        auto gradient_matrix_bias = layer.biases;
        gradient_matrix_bias *= layer.gradients;

        transform_many(adam, gradient_matrix_bias, layer.loss_biases_m, layer.loss_biases_v, layer.biases);
//        transformer3{gradient_matrix_bias, layer.loss_biases, layer.biases}.apply(rmsprop);


        if constexpr (layer_number < layer_count - 1){
            update_weights_layers<layer_number + 1>();
        }
    }


    template <size_t layer_number = 1>
    void random_initialize_layers(const std::function<double()>& random_gen){
        auto& layer = get_layer<layer_number>();
        layer.weights.generate(random_gen);
        layer.biases.generate(random_gen);

        if constexpr (layer_number < layer_count - 1){
            random_initialize_layers<layer_number + 1>(random_gen);
        }
    }
};

#endif //DIGIT_RECOGNIZER_NET_HH
