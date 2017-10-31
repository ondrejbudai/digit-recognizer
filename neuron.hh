#ifndef DIGIT_RECOGNIZER_NEURON_HH
#define DIGIT_RECOGNIZER_NEURON_HH

#include <unordered_map>

struct step_function {
    static double apply(double input) {
        return 1. ? input >= 0. : 0.;
    }
};

struct connection {

};

class neuron {
public:
    virtual double output() = 0;
};

template<typename Transfer_function> class hidden_neuron : public neuron {
public:
    void update(){
        double input = 0;
        for(auto [input_neuron, weight]: m_connections){
            input += input_neuron->output() * weight;
        }

        input += m_bias;
        m_output = Transfer_function::apply(input);
    }

    void add_input(neuron& input_neuron, double weight){
        m_connections[&input_neuron] = weight;
    }

    void set_bias(double bias){
        m_bias = bias;
    }

    double output() override {
        return m_output;
    }

private:
    double m_bias = 0;
    std::unordered_map<neuron*, double> m_connections;
    double m_output = 0;

};

class input_neuron : public neuron {
public:
    double output() override {
        return m_value;
    }

    void set_value(double value){
        m_value = value;
    }
private:
    double m_value = 0;
};

#endif //DIGIT_RECOGNIZER_NEURON_HH
