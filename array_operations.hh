#ifndef DIGIT_RECOGNIZER_ARRAY_OPERATIONS_HH
#define DIGIT_RECOGNIZER_ARRAY_OPERATIONS_HH

#include <cassert>
#include <functional>

template<size_t height, size_t width, typename Type = double>
class matrix_t {
public:
    using Type_ = Type;
    constexpr size_t get_height(){
        return height;
    }
    constexpr size_t get_width(){
        return width;
    }
    Type& operator[](size_t index){
        static_assert(width == 1);
        return m_data[index][0];
    }

    const Type& operator[](size_t index) const {
        static_assert(width == 1);
        return m_data[index][0];
    }

    Type& at(size_t row, size_t col) {
        return m_data[row][col];
    }

    const Type& at(size_t row, size_t col) const {
        return m_data[row][col];
    }

    matrix_t<height, width, Type>& apply(Type(*func)(Type)) {
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++ col){
                m_data[row][col] = func(m_data[row][col]);
            }
        }

        return *this;
    };

    void generate(const std::function<double()>& random_gen){
            for(size_t col = 0; col < width; ++ col){
        for(size_t row = 0; row < height; ++row){
                m_data[row][col] = random_gen();
            }
        }
    }

    void operator+=(const matrix_t<height, width, Type>& b){
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++ col){
                m_data[row][col] += b.m_data[row][col];
            }
        }
    }

    void operator+=(Type value){
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++ col){
                m_data[row][col] += value;
            }
        }
    }

    void operator-=(const matrix_t<height, width, Type>& b){
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++ col){
                m_data[row][col] -= b.m_data[row][col];
            }
        }
    }

    void operator-=(Type value){
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++ col){
                m_data[row][col] -= value;
            }
        }
    }

    void operator*=(const matrix_t<height, width, Type>& b){
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++col){
                m_data[row][col] *= b.m_data[row][col];
            }
        }
    }

    void operator*=(Type value){
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++ col){
                m_data[row][col] *= value;
            }
        }
    }

    void operator/=(const matrix_t<height, width, Type>& b){
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++col){
                m_data[row][col] /= b.m_data[row][col];
            }
        }
    }

    void operator/=(Type value){
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++ col){
                m_data[row][col] /= value;
            }
        }
    }

    Type sum() const {
        Type sum = 0;
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++col){
                sum += m_data[row][col];
            }
        }

        return sum;
    }

    Type max() const {
        Type max = 0;
        for(size_t row = 0; row < height; ++row){
            for(size_t col = 0; col < width; ++col){
                max = std::max(m_data[row][col], max);
            }
        }
        return max;
    }

    void print() const {
        Type max = 0;
            for(size_t col = 0; col < width; ++col){
        for(size_t row = 0; row < height; ++row){
                std::cout << m_data[row][col] << std::endl;
            }
        }
    }

private:
    std::array<std::array<Type, width>, height> m_data{};
};

template<size_t size, typename Type = double>
using vector_t = matrix_t<size, 1, Type>;

template<size_t height, size_t width, typename Type = double>
struct transposed_matrix_t {
public:
    explicit transposed_matrix_t(matrix_t<width, height, Type>& origin) : data{origin} {}

    Type& at(size_t row, size_t col) {
        return data.at(col, row);
    }

    const Type& at(size_t row, size_t col) const {
        return data.at(col, row);
    }

    matrix_t<height, width, Type>& apply(Type(*func)(Type)) {
        return data.apply(func);
    };
private:
    matrix_t<width, height, Type>& data;
};

template<size_t height, size_t width, typename Type>
transposed_matrix_t<width, height, Type> transpose(matrix_t<height, width, Type>& origin){
    return transposed_matrix_t<width, height, Type>{origin};
}

//TODO: change all other templates
template<template<size_t RowsA, size_t ColsA, typename Type> typename MatrixA, template<size_t ColsA, size_t ColsB, typename Type> typename MatrixB, size_t RowsA, size_t ColsA, size_t ColsB, typename Type>
matrix_t<RowsA, ColsB, Type> matrix_multiply(const MatrixA<RowsA, ColsA, Type>& a, const MatrixB<ColsA, ColsB, Type>& b) {
    matrix_t<RowsA, ColsB, Type> result;
    for(size_t i = 0; i < RowsA; ++i){
        for(size_t j = 0; j < ColsB; ++j){
            for(size_t k = 0; k < ColsA; ++k) {
                result.at(i, j) += a.at(i,k) * b.at(k, j);
            }
        }
    }
    return result;
}

void matrix_multiplication_test() {
    matrix_t<2, 4> a;
    matrix_t<4, 3> b;

    a.at(0, 0) = 1;
    a.at(0, 1) = 3;
    a.at(0, 2) = 5;
    a.at(0, 3) = 7;

    a.at(1, 0) = 2;
    a.at(1, 1) = 4;
    a.at(1, 2) = 6;
    a.at(1, 3) = 8;

    b.at(0, 0) = 1;
    b.at(1, 0) = 2;
    b.at(2, 0) = 3;
    b.at(3, 0) = 4;

    b.at(0, 1) = 8;
    b.at(1, 1) = 7;
    b.at(2, 1) = 6;
    b.at(3, 1) = 5;

    b.at(0, 2) = 9;
    b.at(1, 2) = 10;
    b.at(2, 2) = 11;
    b.at(3, 2) = 12;

    matrix_t<2, 3> c;
    c = matrix_multiply(a, b);

    assert(c.at(0, 0) == 50);
    assert(c.at(0, 1) == 94);
    assert(c.at(0, 2) == 178);

    assert(c.at(1, 0) == 60);
    assert(c.at(1, 1) == 120);
    assert(c.at(1, 2) == 220);

    matrix_t<2, 2> test;
    test.at(1, 0) = 66;
    assert(test.at(1,0) == 66);
    assert(transpose(test).at(0,1) == 66);

    matrix_t<2, 4> a1;
    matrix_t<3, 4> b1;

    a1.at(0, 0) = 1;
    a1.at(0, 1) = 3;
    a1.at(0, 2) = 5;
    a1.at(0, 3) = 7;

    a1.at(1, 0) = 2;
    a1.at(1, 1) = 4;
    a1.at(1, 2) = 6;
    a1.at(1, 3) = 8;

    b1.at(0, 0) = 1;
    b1.at(0, 1) = 2;
    b1.at(0, 2) = 3;
    b1.at(0, 3) = 4;

    b1.at(1, 0) = 8;
    b1.at(1, 1) = 7;
    b1.at(1, 2) = 6;
    b1.at(1, 3) = 5;

    b1.at(2, 0) = 9;
    b1.at(2, 1) = 10;
    b1.at(2, 2) = 11;
    b1.at(2, 3) = 12;

    matrix_t<2, 3> c1;
    c1 = matrix_multiply(a1, transpose(b1));

    assert(c1.at(0, 0) == 50);
    assert(c1.at(0, 1) == 94);
    assert(c1.at(0, 2) == 178);

    assert(c1.at(1, 0) == 60);
    assert(c1.at(1, 1) == 120);
    assert(c1.at(1, 2) == 220);
}

template<size_t Height, size_t Width, typename Type>
struct transformer2 {

    using Matrix = matrix_t<Height, Width, Type>;
    Matrix& a;
    Matrix& b;
    transformer2(Matrix& a_, Matrix& b_) : a{a_}, b{b_}{}

    void apply(void(*func)(Type&, Type&)){
        for(size_t row = 0; row < Height; ++row){
            for(size_t col = 0; col < Width; ++col){
                func(a.at(row, col), b.at(row, col));
            }
        }
    }
};

template<size_t Height, size_t Width, typename Type>
struct transformer3 {

    using Matrix = matrix_t<Height, Width, Type>;
    Matrix& a;
    Matrix& b;
    Matrix& c;
    transformer3(Matrix& a_, Matrix& b_, Matrix& c_) : a{a_}, b{b_}, c{c_}{}

    void apply(void(*func)(Type&, Type&, Type&)){
        for(size_t row = 0; row < Height; ++row){
            for(size_t col = 0; col < Width; ++col){
                func(a.at(row, col), b.at(row, col), c.at(row, col));
            }
        }
    }
};

#endif //DIGIT_RECOGNIZER_ARRAY_OPERATIONS_HH
