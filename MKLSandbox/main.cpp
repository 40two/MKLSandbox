#include "mkl_interface.hpp"
#include <iostream>

int main() {
  double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

  std::cout << "Before:" << std::endl;
  for(auto &&val : a) std::cout << val << " ";
  std::cout << std::endl;
  for(auto &&val : b) std::cout << val << " ";
  std::cout << std::endl;
  mkl::dswap(sizeof(a)/sizeof(double), a, 1, b, 1);
  std::cout << "After:" << std::endl;
  for (auto &&val : a) std::cout << val << " ";
  std::cout << std::endl;
  for (auto &&val : b) std::cout << val << " ";
  std::cout << std::endl;

  system("PAUSE");

}