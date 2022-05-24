#pragma once

#include <math.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

static void print_vector(std::string info, std::vector<double> vec,
                         size_t precision, std::ostream &out, size_t len) {
  std::ios_base::fmtflags f(out.flags());  // get flags
  if (!info.empty()) {
    out << info << " ";
  }
  out << "{";
  out << std::fixed;
  const auto delim = ", ";
  for (size_t i = 0; i < len; i++) {
    out << std::setprecision(precision);
    out << vec[i] << delim;
  }
  out << "}\n";

  out.flags(f);  // reset flags
}

static void print_double(std::string info, double val, size_t precision,
                         std::ostream &out) {
  std::ios_base::fmtflags f(out.flags());  // get flags
  if (!info.empty()) {
    out << info << " ";
  }
  out << std::fixed;
  out << std::setprecision(precision);
  out << val << std::endl;
  ;
  out.flags(f);  // reset flags
}

std::string double_to_string(double a, size_t precision) {
  if (abs(a) < 0.000001) a = 0.;
  std::stringstream ss;
  ss << std::setprecision(precision) << a;
  return ss.str();
}

static bool compare_vectors(std::vector<double> &v1, std::vector<double> &v2,
                            size_t precision) {
  for (size_t i = 0; i < v1.size(); i++) {
    auto s1 = double_to_string(v1[i], precision);
    auto s2 = double_to_string(v2[i], precision);
    if (s1 != s2) {
      std::cout << "  mismatch at index " << i << std::endl;
      std::cout << "  " << s1 << " " << s2 << std::endl;
      return false;
    }
  }
  return true;
}
