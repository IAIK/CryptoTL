#pragma once

#include "seal/seal.h"

// #define SERVER_ONLY

class Server {
 public:
  static constexpr size_t IN_SIZE = 768;

 private:
  // Filenames
  static constexpr const char *PARAM_FILE = "param";
  static constexpr const char *GK_FILE = "gk";
  static constexpr const char *RK_FILE = "rk";
  static constexpr const char *CIPHER_FILE = "cipher_in";
  static constexpr const char *RESULT_FILE = "cipher_out";

  // network parameters (NLP)
  static constexpr uint64_t LEVELS = 6;
  static constexpr size_t CONV1_NUM_FILTERS = 1;
  static constexpr size_t CONV1_FILTER_SIZE = 9;
  static constexpr size_t DENSE1_IN = 768;
  static constexpr size_t DENSE1_OUT = 768;
  static constexpr uint64_t DENSE1_BSGS_N1 = 32;
  static constexpr uint64_t DENSE1_BSGS_N2 = 24;
  static constexpr size_t AVG_FILTER_SIZE = 3;
  static constexpr size_t DENSE2_IN = 766;
  static constexpr uint64_t DENSE2_BSGS_N1 = 383;
  static constexpr uint64_t DENSE2_BSGS_N2 = 2;
#if defined(SERVER_ONLY)
  static constexpr size_t OUT_SIZE = 1;
#else
  static constexpr size_t OUT_SIZE = 766;
#endif

  static constexpr bool NORMALIZE_SET_SCALE = false;

  typedef std::vector<double> double_vector;
  typedef std::vector<std::vector<double>> double_matrix;

  std::shared_ptr<seal::SEALContext> context;
  seal::Evaluator evaluator;
  std::unique_ptr<seal::CKKSEncoder> ckks_encoder;

  seal::GaloisKeys galois_keys;
  seal::RelinKeys relin_keys;

  uint64_t slots;
  uint64_t plain_mod;
  double scale;
  size_t bitscale;
  uint64_t current_level = 0;

  bool use_bsgs = false;
  bool gk_set = false;
  bool relin_set = false;

  uint64_t bsgs_n1;
  uint64_t bsgs_n2;

  std::vector<double_vector> conv1d_filter;
  double_matrix dense1_w;
  double_vector dense1_b;
  double_matrix dense2_w;
  double_vector dense2_b;

  void CKKS_diagonal(seal::Ciphertext &in_out, const double_matrix &mat);
  void CKKS_babystep_giantstep(seal::Ciphertext &in_out,
                               const double_matrix &mat);

  void plain_mat(double_vector &out, const double_vector &in,
                 const double_matrix &mat, size_t outsize);

  // neural network:
  void dense(seal::Ciphertext &in_out, const double_matrix &mat,
             const double_vector &bias, bool skip_rescale = false);
  void conv1d(seal::Ciphertext &in_out, size_t in_size,
              const std::vector<double_vector> &filter);
  void conv2d(seal::Ciphertext &in_out, size_t in_size,
              const std::vector<double_matrix> &kernel);
  // void relu2(seal::Ciphertext &in_out);
  void relu3(seal::Ciphertext &in_out);
  // void relu4(seal::Ciphertext &in_out);
  void avgpool(seal::Ciphertext &in_out, uint64_t filter_size,
               uint64_t out_length);  // stride = 1

  void dense_plain(double_vector &out, const double_vector &in,
                   const double_matrix &mat, const double_vector &bias);
  void conv1d_plain(double_vector &out, const double_vector &in,
                    const std::vector<double_vector> &filter);
  void conv2d_plain(double_matrix &out, const double_matrix &in,
                    const double_matrix &kernel);
  // void relu2_plain(double_vector &in_out);
  void relu3_plain(double_vector &in_out);
  // void relu4_plain(double_vector &in_out);
  void avgpool_plain(double_vector &out, const double_vector &in,
                     uint64_t filter_size, uint64_t out_length);  // stride = 1

  static double parse(std::string);
  void read_conv1d(std::ifstream &, size_t num_filters, size_t filter_size,
                   std::vector<double_vector> &) const;
  void read_dense(std::ifstream &, size_t in_size, size_t out_size,
                  double_matrix &, double_vector &) const;

 public:
  Server(std::shared_ptr<seal::SEALContext> context, size_t bitscale);
  Server() = delete;
  ~Server() = default;
  Server(const Server &) = delete;
  Server &operator=(const Server &) = delete;

  void reset_level();
  uint64_t get_level() const;
  void activate_bsgs(bool use_bsgs);
  static std::shared_ptr<seal::SEALContext> context_from_file();

  uint64_t keys_from_file(bool relin = true);

  void print_parameters();

  void set_gk(seal::GaloisKeys &);
  void set_rk(seal::RelinKeys &);

  void cipher_from_file(seal::Ciphertext &cipher);
  void cipher_to_file(seal::Ciphertext &cipher);

  // neural network
  void network(seal::Ciphertext &in_out);
  void network_plain(double_vector &out, const double_vector &in);

  void read_weights(const std::string &filename);

  static std::vector<int> required_gk_indices(bool use_bsgs);
  static uint64_t required_levels();
  size_t get_out_size();
};
