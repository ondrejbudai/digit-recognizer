#ifndef DIGIT_RECOGNIZER_NET_HH
#define DIGIT_RECOGNIZER_NET_HH

#include <array>
#include <tuple>

template<size_t neuron_count_in_previous_layer> class nneuron {
public:
    nneuron(){
        weights.fill(0);
    }
    std::array<double, neuron_count_in_previous_layer> weights;
    double output;
    double bias;
};

template<size_t neuron_count_, size_t neuron_count_in_previous_layer>
class layer {
public:
    static constexpr size_t neuron_count = neuron_count_;
    std::array<nneuron<neuron_count_in_previous_layer>, neuron_count> neurons;
    template<size_t neuron_number> auto& get_neuron(){
        static_assert(neuron_number < neuron_count);
        return neurons[neuron_number];
    }
};

template <size_t...> struct size_tuple{};

template< std::size_t I, class T >
struct size_tuple_element;

// recursive case
template< std::size_t I, size_t Head, size_t... Tail >
struct size_tuple_element<I, size_tuple<Head, Tail...>>
    : size_tuple_element<I-1, size_tuple<Tail...>> { };

// base case
template< size_t Head, size_t... Tail >
struct size_tuple_element<0,size_tuple<Head, Tail...>> {
    static constexpr size_t value = Head;
};

template <size_t ...> struct holder;

template <class T, size_t... Ts> struct foobase;

template <std::size_t... I, size_t... Ts>
struct foobase<std::index_sequence<I...>, Ts...> {
    using bar = holder<0, size_tuple_element<I, size_tuple<Ts...>>::value...>;
};

template <size_t... Ts> struct foo
    : foobase<std::make_index_sequence<sizeof...(Ts) - 1>, Ts...>
{
};

template<size_t ...Args1> struct zip {
    template<class> struct with;

    template<size_t ...Args2>
    struct with<holder<Args2...>> {
        using type = std::tuple<layer<Args1, Args2>...>;
    };
};

template<size_t... args> class net {
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

    void evaluate(){
        evaluate_layers();
    }

private:
    static constexpr size_t layer_count = sizeof...(args);
    typename zip<args...>::template with<typename foo<args...>::bar>::type layers;

    template<size_t layer_number> static constexpr size_t get_neuron_count_at_layer(){
        return std::tuple_element<layer_number, decltype(layers)>::type::neuron_count;
    }

    template<size_t layer_number> auto& get_layer(){
        return std::get<layer_number>(layers);
    }

    template<size_t layer_number = 1> void evaluate_layers(){
        static_assert(layer_number > 0, "First layer cannot be evaluated!");
        evaluate_neurons<layer_number>();
        if constexpr(layer_number < layer_count - 1){
            evaluate_layers<layer_number+1>();
        }
    }

    template<size_t layer_number, size_t neuron_number = 0> void evaluate_neurons(){

        double value = get_layer<layer_number>().neurons[neuron_number].bias;

        for(size_t input_number = 0; input_number < get_neuron_count_at_layer<layer_number-1>(); ++input_number){
            value += get_layer<layer_number - 1>().neurons[input_number].output * get_layer<layer_number>().neurons[neuron_number].weights[input_number];
        }

        get_layer<layer_number>().neurons[neuron_number].output = value >= 0 ? 1 : 0;

        if constexpr(neuron_number < get_neuron_count_at_layer<layer_number>() - 1){
            evaluate_neurons<layer_number, neuron_number + 1>();
        }


    };
};

#endif //DIGIT_RECOGNIZER_NET_HH
