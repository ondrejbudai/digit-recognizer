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
    static void derivative(vector_t<Values, Type>& output, const vector_t<Values, Type>& above){
        output.apply([](auto value){return value > 0 ? decltype(value)(1.) : 0.;});
        output *= above;
    }
};

struct hyperbolic_tangent {
    template<size_t Values, typename Type>
    static void eval(vector_t<Values, Type>& input){
        input.apply(std::tanh);
    }

    template<size_t Values, typename Type>
    static void derivative(vector_t<Values, Type>& output, const vector_t<Values, Type>& above){
        output.apply([](auto value){return 1 - value * value;});
        output *= above;
    }
};

struct softmax {
    template<size_t Values, typename Type>
    static void eval(vector_t<Values, Type>& input){
        auto maximum = input.max();
        input -= maximum;
        input.apply([](auto value){return std::exp(value);});
        auto sum = input.sum();
        input /= sum;
    }

    template<size_t Values, typename Type>
    static void derivative(vector_t<Values, Type>& output, const vector_t<Values, Type>& above){
        auto delta = output;
        delta *= above;
        auto sum = delta.sum();
        output *= sum;
        delta -= output;
        output = delta;

    }
};

//struct soft_max {
//    static double eval(vector input){
//        maximum = max_vector(input);
//        exps = exp(max_vector - maximum);
//
//        return exps / exps.sum();
//    }
//};

//template<size_t neuron_count_in_previous_layer>
//struct neuron {
//    neuron(){
//        weights.fill(0);
//        deltas.fill(0);
//    }
//    std::array<double, neuron_count_in_previous_layer> weights{};
//    std::array<double, neuron_count_in_previous_layer> deltas{};
//    std::array<double, neuron_count_in_previous_layer> current_deltas{};
//    double output = 0.;
//    double bias = 0.;
//    double bias_current_delta = 0.;
//    double biasdelta = 0.;
//    double gradient = 0.;
//    bool enabled = true;
//};

template<size_t neuron_count_, size_t neuron_count_in_previous_layer>
struct layer {
    static constexpr size_t neuron_count = neuron_count_;
    vector_t<neuron_count, double> outputs;
    vector_t<neuron_count, double> biases;
    vector_t<neuron_count, double> loss_biases;
    vector_t<neuron_count, double> biases_deltas;
    vector_t<neuron_count, double> gradients;
    matrix_t<neuron_count_in_previous_layer, neuron_count, double> weights;
    matrix_t<neuron_count_in_previous_layer, neuron_count, double> loss;
    matrix_t<neuron_count_in_previous_layer, neuron_count, double> weights_deltas;
//    std::array<neuron<neuron_count_in_previous_layer>, neuron_count> weights;

//    std::array<double, neuron_count> outputs{};
//    std::array<double, neuron_count> biases{};
//    std::array<double, neuron_count> gradients{};
//    std::array<double, neuron_count> bias_current_deltas{};
//    std::array<double, neuron_count> biasdeltas{};

//    template<size_t neuron_number>
//    auto& get_neuron(){
//        static_assert(neuron_number < neuron_count);
//        return neurons[neuron_number];
//    }
//
//    template<size_t neuron_number>
//    const auto& get_neuron() const{
//        static_assert(neuron_number < neuron_count);
//        return neurons[neuron_number];
//    }
//
//    auto& get_neuron(size_t neuron_number){
//        return neurons[neuron_number];
//    }
//    const auto& get_neuron(size_t neuron_number) const{
//        return neurons[neuron_number];
//    }
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
    vector_t<output_count, double> target_values;

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
//            softmax::eval(layer.outputs);
            ActivationFunction::eval(layer.outputs);
        }

//        layer.outputs.print();

        if constexpr(layer_number < layer_count - 1){
            evaluate_layers<layer_number+1>();
        }
    }

    void calc_gradient_output_layer(){
//        std::cout << "gradiento" << std::endl;
        auto& output_layer = get_layer<layer_count - 1>();

        // squared error
        target_values *= -1;
        target_values += output_layer.outputs;
        //crossentropy

//        auto denominators = output_layer.outputs;
//        denominators.apply([](auto value){return std::max(value - value * value, decltype(value)(1e-11));});
//        target_values *= -1;
//        target_values += output_layer.outputs;
//        target_values /= denominators;

        output_layer.gradients = output_layer.outputs;
//        softmax::derivative(output_layer.gradients, target_values);
        ActivationFunction::derivative(output_layer.gradients, target_values);
//        output_layer.gradients.print();
    }

    template<size_t layer_number = layer_count - 2>
    void calc_gradient_hidden_layers(){
//        std::cout << "gradientl" << layer_number << std::endl;
        auto& layer = get_layer<layer_number>();
        auto& next_layer = get_layer<layer_number + 1>();

        auto a = matrix_multiply(next_layer.weights, next_layer.gradients);
        layer.gradients = layer.outputs;
        ActivationFunction::derivative(layer.gradients, a);

//        layer.gradients.print();

        if constexpr (layer_number > 1){
            calc_gradient_hidden_layers<layer_number - 1>();
        }
    }

    template<size_t layer_number = 1>
    void update_weights_layers(){
//        std::cout << "weightsl" << layer_number << std::endl;
        auto& layer = get_layer<layer_number>();
        auto& prev_layer = get_layer<layer_number-1>();

        constexpr auto decay = 0.9;
        constexpr auto eps = 1e-8;

        auto gradient_matrix = matrix_multiply(prev_layer.outputs, transpose(layer.gradients));
//        auto gradient_matrix2 = gradient_matrix;
//        gradient_matrix2.apply([](auto value){return value * value;});
//        gradient_matrix2 *= 1 - decay;
//        layer.loss *= decay;
//        layer.loss += gradient_matrix2;
//        auto b = layer.loss;
//        b.apply([](auto value){return std::sqrt(value);});
//        b += eps;
//
//        gradient_matrix /= b;
//
//        gradient_matrix *= 0.001;
////        layer.weights_deltas *= 0.8;
////        gradient_matrix += layer.weights_deltas;
//        layer.weights -= gradient_matrix;
//        layer.weights_deltas = gradient_matrix;

//        layer.weights.print();


        transformer3{gradient_matrix, layer.loss, layer.weights}.apply([](auto& gradient, auto& loss, auto& weight){
            loss = decay * loss + (1 - decay) * gradient * gradient;
            weight -= 0.00005 * gradient / (std::sqrt(loss) + eps);
        });

        auto gradient_matrix_bias = layer.biases;
        gradient_matrix_bias *= layer.gradients;

//        auto gradient_matrix_bias2 = gradient_matrix_bias;
//        gradient_matrix_bias2.apply([](auto value){return value * value;});
//        gradient_matrix_bias2 *= 1 - decay;
//        layer.loss_biases *= decay;
//        layer.loss_biases += gradient_matrix_bias2;
//        auto c = layer.loss_biases;
//        c.apply([](auto value){return std::sqrt(value);});
//        c += eps;
//
//
//        gradient_matrix_bias /= c;
//
//
//        gradient_matrix_bias *= 0.001;
////        layer.biases_deltas *= 0.8;
////        gradient_matrix_bias += layer.biases_deltas;
//        layer.biases -= gradient_matrix_bias;
//        layer.biases_deltas = gradient_matrix_bias;

        transformer3{gradient_matrix_bias, layer.loss_biases, layer.biases}.apply([](auto& gradient, auto& loss, auto& weight){
            loss = decay * loss + (1 - decay) * gradient * gradient;
            weight -= 0.00005 * gradient / (std::sqrt(loss) + eps);
        });

//        layer.biases.print();

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
