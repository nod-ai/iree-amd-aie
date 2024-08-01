// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_RUNTIME_MACROS_H
#define IREE_AIE_RUNTIME_MACROS_H

#ifdef _WIN32
#ifndef IREE_AIE_RUNTIME_EXPORT
#ifdef iree_aie_runtime_EXPORTS
// We are building this library
#define IREE_AIE_RUNTIME_EXPORT __declspec(dllexport)
#else
// We are using this library
#define IREE_AIE_RUNTIME_EXPORT __declspec(dllimport)
#endif  // iree_aie_runtime_EXPORTS
#endif  // IREE_AIE_RUNTIME_EXPORT
#else
// Non-windows: use visibility attributes.
#define IREE_AIE_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#define STRINGIFY_ENUM_CASE(case_) \
  case (case_):                    \
    return #case_;

#define OSTREAM_OP_DECL(O_TYPE, TYPE) \
  O_TYPE& operator<<(O_TYPE& os, const TYPE& s);

#define OSTREAM_OP_DEFN(O_TYPE, TYPE)                \
  O_TYPE& operator<<(O_TYPE& os, const TYPE& s) {    \
    os << mlir::iree_compiler::AMDAIE::to_string(s); \
    return os;                                       \
  }

#define TO_STRING_DECL(TYPE) std::string to_string(const TYPE& t);

#define BOTH_OSTREAM_OP(OSTREAM_OP_, TYPE) \
  OSTREAM_OP_(std::ostream, TYPE)          \
  OSTREAM_OP_(llvm::raw_ostream, TYPE)

#define STRINGIFY_2TUPLE_STRUCT(Type, first, second) \
  std::string to_string(const Type& t) {             \
    std::string s = #Type "(" #first ": ";           \
    s += to_string(t.first);                         \
    s += ", " #second ": ";                          \
    s += to_string(t.second);                        \
    s += ")";                                        \
    return s;                                        \
  }

#define STRINGIFY_3TUPLE_STRUCT(Type, first, second, third) \
  std::string to_string(const Type& t) {                    \
    std::string s = #Type "(" #first ": ";                  \
    s += to_string(t.first);                                \
    s += ", " #second ": ";                                 \
    s += to_string(t.second);                               \
    s += ", " #third ": ";                                  \
    s += to_string(t.third);                                \
    s += ")";                                               \
    return s;                                               \
  }

#define STRINGIFY_4TUPLE_STRUCT(Type, first, second, third, fourth) \
  std::string to_string(const Type& t) {                            \
    std::string s = #Type "(" #first ": ";                          \
    s += to_string(t.first);                                        \
    s += ", " #second ": ";                                         \
    s += to_string(t.second);                                       \
    s += ", " #third ": ";                                          \
    s += to_string(t.third);                                        \
    s += ", " #fourth ": ";                                         \
    s += to_string(t.fourth);                                       \
    s += ")";                                                       \
    return s;                                                       \
  }

#define STRINGIFY_5TUPLE_STRUCT(Type, first, second, third, fourth, fifth) \
  std::string to_string(const Type& t) {                                   \
    std::string s = #Type "(" #first ": ";                                 \
    s += to_string(t.first);                                               \
    s += ", " #second ": ";                                                \
    s += to_string(t.second);                                              \
    s += ", " #third ": ";                                                 \
    s += to_string(t.third);                                               \
    s += ", " #fourth ": ";                                                \
    s += to_string(t.fourth);                                              \
    s += ", " #fifth ": ";                                                 \
    s += to_string(t.fifth);                                               \
    s += ")";                                                              \
    return s;                                                              \
  }

#define ASSERT_STANDARD_LAYOUT(p)             \
  static_assert(std::is_standard_layout_v<p>, \
                #p " is meant to be a standard layout type")

#define TUPLE_LIKE_STRUCT_RELATIONAL_OPS(T)    \
  bool operator<(const T& rhs) const {         \
    return TupleType(*this) < TupleType(rhs);  \
  }                                            \
  bool operator==(const T& rhs) const {        \
    return TupleType(*this) == TupleType(rhs); \
  }                                            \
  bool operator!=(const T& rhs) const { return !(*this == rhs); }

// https://stackoverflow.com/a/20170989/9045206
// useful for debugging templates
// std::cout << type_name<decltype(t)>() << '\n';
template <class T>
constexpr std::string_view type_name() {
  using namespace std;
#ifdef __clang__
  string_view p = __PRETTY_FUNCTION__;
  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
  string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#else
  return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
  string_view p = __FUNCSIG__;
  return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

#endif  // IREE_MACROS_H
