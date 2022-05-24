#pragma once

#include "seal/seal.h"

class Client {
 public:
  static constexpr size_t OUT_SIZE = 1;

 private:
  // Filenames
  static constexpr const char *SK_FILE = "sk";
  static constexpr const char *PARAM_FILE = "param";
  static constexpr const char *GK_FILE = "gk";
  static constexpr const char *RK_FILE = "rk";
  static constexpr const char *CIPHER_FILE = "cipher_in";
  static constexpr const char *RESULT_FILE = "cipher_out";

  // network parameters (NLP)
  static constexpr size_t DENSE1_IN = 766;
  static constexpr size_t DENSE1_OUT = 766;
  static constexpr size_t MAX_FILTER_SIZE = 3;
  static constexpr size_t DENSE2_IN = 764;

  typedef std::vector<double> double_vector;
  typedef std::vector<std::vector<double>> double_matrix;

  std::shared_ptr<seal::SEALContext> context;
  seal::KeyGenerator keygen;

  seal::GaloisKeys galois_keys;
  seal::SecretKey secret_key;
  seal::RelinKeys relin_keys;

  seal::Encryptor encryptor;
  seal::Evaluator evaluator;
  seal::Decryptor decryptor;
  std::unique_ptr<seal::CKKSEncoder> ckks_encoder;

  uint64_t slots;
  uint64_t plain_mod;
  seal::scheme_type scheme;
  double scale;
  size_t bitscale;

  double_matrix dense1_w;
  double_vector dense1_b;
  double_matrix dense2_w;
  double_vector dense2_b;

  std::vector<double_vector> dataset;
  std::vector<int> labels;

  static double parse_double(std::string);
  static int parse_int(std::string);
  void read_conv1d(std::ifstream &, size_t num_filters, size_t filter_size,
                   std::vector<double_vector> &) const;
  void read_dense(std::ifstream &, size_t in_size, size_t out_size,
                  double_matrix &, double_vector &) const;

  void plain_mat(double_vector &out, const double_vector &in,
                 const double_matrix &mat, size_t outsize);
  void sigmoid(double_vector &in_out) const;
  void dense(double_vector &out, const double_vector &in,
             const double_matrix &mat, const double_vector &bias);
  void conv1d(double_vector &out, const double_vector &in,
              const std::vector<double_vector> &filter);
  void relu(double_vector &in_out);
  void avgpool(double_vector &out, const double_vector &in,
               uint64_t filter_size, uint64_t out_length);  // stride = 1
  void maxpool(double_vector &out, const double_vector &in,
               uint64_t filter_size, uint64_t out_length);  // stride = 1

 public:
  Client(std::shared_ptr<seal::SEALContext>, size_t bitscale,
         bool to_file = false);
  Client(std::shared_ptr<seal::SEALContext>, seal::SecretKey &,
         size_t bitscale);
  ~Client() = default;
  Client(const Client &) = delete;
  Client &operator=(const Client &) = delete;

  static std::shared_ptr<seal::SEALContext> create_CKKS_context(
      size_t mod_degree, size_t bitscale, size_t fixscale, size_t levels,
      bool to_file = false);

  static std::shared_ptr<seal::SEALContext> context_from_file();

  int get_noise(seal::Ciphertext &, bool print = false);
  void print_parameters();
  uint64_t get_slots() const;

  seal::GaloisKeys &get_galois_keys();
  seal::RelinKeys get_relin_keys();

  void create_gk(std::vector<int> &gks, bool to_file = false);

  void cipher_from_file(seal::Ciphertext &cipher);

  static void sk_from_file(seal::SecretKey &,
                           std::shared_ptr<seal::SEALContext> &);

  void encrypt(seal::Ciphertext &, double_vector &);
  void encrypt_to_file(double_vector &);
  void decrypt(double_vector &, seal::Ciphertext &);

  // the added layers
  void network(double_vector &in_out);

  void read_inputs(const std::string &filename, bool with_label);
  int get_num_datasets() const;
  double_vector get_dataset(size_t index) const;
  int get_label(size_t index) const;

  void read_weights(const std::string &filename);
};
