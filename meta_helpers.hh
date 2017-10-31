#ifndef DIGIT_RECOGNIZER_META_HELPERS_HH
#define DIGIT_RECOGNIZER_META_HELPERS_HH

#include <utility>

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

template <class T, size_t... Ts> struct remove_last_from_size_typle_base;

template <std::size_t... I, size_t... Ts>
struct remove_last_from_size_typle_base<std::index_sequence<I...>, Ts...> {
using type = size_tuple<0, size_tuple_element<I, size_tuple<Ts...>>::value...>;
};

template <size_t... Ts> struct remove_last_from_size_typle
    : remove_last_from_size_typle_base<std::make_index_sequence<sizeof...(Ts) - 1>, Ts...>
{
};

template<typename ValueType, template<typename ...> typename Tuple, template<ValueType ...>typename Pair, ValueType ...Args1> struct zip {
    template<class> struct with;

    template<ValueType ...Args2>
    struct with<size_tuple<Args2...>> {
        using type = Tuple<Pair<Args1, Args2>...>;
    };
};

#endif //DIGIT_RECOGNIZER_META_HELPERS_HH
