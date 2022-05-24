#include "Client.h"

#include <assert.h>

#include <fstream>
#include <iostream>

using namespace seal;

//----------------------------------------------------------------

Client::Client(std::shared_ptr<seal::SEALContext> con, size_t sc, bool to_file)
    : context(con),
      keygen(*context),
      secret_key(keygen.secret_key()),
      encryptor(*context, secret_key),
      evaluator(*context),
      decryptor(*context, secret_key),
      plain_mod(0),
      scheme(context->key_context_data()->parms().scheme()),
      scale(pow(2.0, sc)),
      bitscale(sc) {
  keygen.create_relin_keys(relin_keys);
  ckks_encoder = std::make_unique<seal::CKKSEncoder>(*context);
  slots = ckks_encoder->slot_count();

  if (to_file) {
    std::ofstream rk;
    rk.open(RK_FILE);
    auto rlk = keygen.create_relin_keys();
    rlk.save(rk);

    std::ofstream sk;
    sk.open(SK_FILE);
    secret_key.save(sk);
  }
}

//----------------------------------------------------------------

Client::Client(std::shared_ptr<seal::SEALContext> con, seal::SecretKey &sk,
               size_t sc)
    : context(con),
      keygen(*context),
      secret_key(sk),
      encryptor(*context, secret_key),
      evaluator(*context),
      decryptor(*context, secret_key),
      plain_mod(0),
      scheme(context->key_context_data()->parms().scheme()),
      scale(pow(2.0, sc)),
      bitscale(sc) {
  keygen.create_relin_keys(relin_keys);
  ckks_encoder = std::make_unique<seal::CKKSEncoder>(*context);
  slots = ckks_encoder->slot_count();
}

//----------------------------------------------------------------

void Client::sk_from_file(seal::SecretKey &seckey,
                          std::shared_ptr<seal::SEALContext> &context) {
  std::ifstream sk;
  sk.open(SK_FILE);
  seckey.load(*context, sk);
}

//----------------------------------------------------------------

void Client::cipher_from_file(seal::Ciphertext &ciph) {
  std::ifstream c;
  c.open(RESULT_FILE);
  ciph.load(*context, c);
}

//----------------------------------------------------------------

std::shared_ptr<seal::SEALContext> Client::create_CKKS_context(
    size_t mod_degree, size_t bitscale, size_t fixscale, size_t levels,
    bool to_file) {
  EncryptionParameters parms(scheme_type::ckks);
  parms.set_poly_modulus_degree(mod_degree);
  sec_level_type sec = sec_level_type::tc128;

  int bitsum = 2 * fixscale;
  std::vector<int> bits;
  bits.reserve(levels + 2);
  bits.push_back(fixscale);
  for (size_t i = 0; i < levels; i++) {
    bits.push_back(bitscale);
    bitsum += bitscale;
  }
  bits.push_back(fixscale);

  if (bitsum > CoeffModulus::MaxBitCount(mod_degree, sec))
    throw std::invalid_argument("Mod to big for security level!");

  parms.set_coeff_modulus(CoeffModulus::Create(mod_degree, bits));

  if (to_file) {
    std::ofstream param;
    param.open(PARAM_FILE);
    parms.save(param);
  }

  return std::make_shared<seal::SEALContext>(parms, true, sec);
}

//----------------------------------------------------------------

std::shared_ptr<seal::SEALContext> Client::context_from_file() {
  EncryptionParameters parms;

  std::ifstream param;
  param.open(PARAM_FILE);
  parms.load(param);

  sec_level_type sec = sec_level_type::tc128;
  return std::make_shared<seal::SEALContext>(parms, true, sec);
}

//----------------------------------------------------------------

int Client::get_noise(seal::Ciphertext &ciph, bool print) {
  if (scheme != scheme_type::bfv) return 0;

  int noise = decryptor.invariant_noise_budget(ciph);
  if (print) std::cout << "noise budget: " << noise << std::endl;
  return noise;
}

//----------------------------------------------------------------

void Client::print_parameters() {
  // Verify parameters
  if (!context) {
    throw std::invalid_argument("context is not set");
  }
  auto &context_data = *context->key_context_data();

  /*
  Which scheme are we using?
  */
  std::string scheme_name;
  switch (context_data.parms().scheme()) {
    case scheme_type::bfv:
      scheme_name = "BFV";
      break;
    case scheme_type::ckks:
      scheme_name = "CKKS";
      break;
    default:
      throw std::invalid_argument("unsupported scheme");
  }
  std::cout << "/" << std::endl;
  std::cout << "| Encryption parameters:" << std::endl;
  std::cout << "|   scheme: " << scheme_name << std::endl;
  std::cout << "|   poly_modulus_degree: "
            << context_data.parms().poly_modulus_degree() << std::endl;

  /*
  Print the size of the true (product) coefficient modulus.
  */
  std::cout << "|   coeff_modulus size: ";
  std::cout << context_data.total_coeff_modulus_bit_count() << " (";
  auto coeff_modulus = context_data.parms().coeff_modulus();
  std::size_t coeff_mod_count = coeff_modulus.size();
  for (std::size_t i = 0; i < coeff_mod_count - 1; i++) {
    std::cout << coeff_modulus[i].bit_count() << " + ";
  }
  std::cout << coeff_modulus.back().bit_count();
  std::cout << ") bits" << std::endl;

  /*
  For the BFV scheme print the plain_modulus parameter.
  */
  if (context_data.parms().scheme() == scheme_type::bfv) {
    std::cout << "|   plain_modulus: "
              << context_data.parms().plain_modulus().value() << std::endl;
  }
  std::cout << "|   scale (bits): " << bitscale << std::endl;

  std::cout << "\\" << std::endl;
}

//----------------------------------------------------------------

void Client::create_gk(std::vector<int> &gks, bool to_file) {
  if (to_file) {
    std::ofstream gk;
    gk.open(GK_FILE);
    auto glk = keygen.create_galois_keys(gks);
    glk.save(gk);
  } else
    keygen.create_galois_keys(gks, galois_keys);
}

//----------------------------------------------------------------

uint64_t Client::get_slots() const { return slots; }

//----------------------------------------------------------------

seal::GaloisKeys &Client::get_galois_keys() { return galois_keys; }

//----------------------------------------------------------------

seal::RelinKeys Client::get_relin_keys() { return relin_keys; }

//----------------------------------------------------------------

void Client::encrypt(Ciphertext &ciph, std::vector<double> &input) {
  Plaintext plain;
  ckks_encoder->encode(input, scale, plain);
  encryptor.encrypt_symmetric(plain, ciph);
}

//----------------------------------------------------------------

void Client::decrypt(std::vector<double> &res, Ciphertext &ciph) {
  Plaintext ptxt;
  decryptor.decrypt(ciph, ptxt);
  ckks_encoder->decode(ptxt, res);
}

//----------------------------------------------------------------

void Client::encrypt_to_file(std::vector<double> &input) {
  std::ofstream c;
  c.open(CIPHER_FILE);

  Plaintext plain;
  ckks_encoder->encode(input, scale, plain);
  auto ct = encryptor.encrypt_symmetric(plain);
  ct.save(c);
}

//----------------------------------------------------------------

double Client::parse_double(std::string in) {
  double out;
  std::stringstream buffer(in);
  buffer >> out;
  if (buffer.bad() || !buffer.eof())
    throw std::runtime_error("Error parsing string to double");
  return out;
}

//----------------------------------------------------------------

int Client::parse_int(std::string in) {
  int out;
  std::stringstream buffer(in);
  buffer >> out;
  if (buffer.bad() || !buffer.eof())
    throw std::runtime_error("Error parsing string to double");
  return out;
}

//----------------------------------------------------------------

void Client::read_inputs(const std::string &filename, bool with_label) {
  std::ifstream file(filename);
  if (!file.is_open()) throw std::runtime_error("Error opening weight file");
  std::string line, tmp;

  std::vector<std::string> elements;
  std::getline(file, line);
  std::stringstream buffer(line);
  while (buffer >> tmp) elements.push_back(tmp);
  if (elements.size() != 6) throw ::std::runtime_error("Wrong file format");
  if (elements[1] != "datasets" || elements[2] != "with" ||
      elements[4] != "features" || elements[5] != "each:")
    throw ::std::runtime_error("Wrong file format");

  int num_datasets = parse_int(elements[0]);
  int features = parse_int(elements[3]);

  dataset.resize(num_datasets, double_vector(features));
  if (with_label) labels.resize(num_datasets, 0);

  for (int i = 0; i < num_datasets; i++) {
    std::string expect_line = "Dataset " + std::to_string(i);
    std::getline(file, line);
    if (line != expect_line) throw ::std::runtime_error("Wrong file format");
    for (int j = 0; j < features; j++) {
      std::getline(file, line);
      dataset[i][j] = parse_double(line);
    }
    if (with_label) {
      expect_line = "Label " + std::to_string(i);
      std::getline(file, line);
      if (line != expect_line) throw ::std::runtime_error("Wrong file format");
      std::getline(file, line);
      labels[i] = parse_int(line);
    }
  }
}

// ----------------------------------------------------------------

int Client::get_num_datasets() const { return dataset.size(); }

//----------------------------------------------------------------

std::vector<double> Client::get_dataset(size_t index) const {
  return dataset[index];
}

//----------------------------------------------------------------

int Client::get_label(size_t index) const { return labels[index]; }

//----------------------------------------------------------------

void Client::sigmoid(double_vector &in_out) const {
  for (auto &it : in_out) {
    double denom = std::exp(-it) + 1;
    it = 1 / denom;
  }
}

//----------------------------------------------------------------

void Client::network(double_vector &in_out) {
#if defined(SERVER_ONLY)
  (void)network;
  sigmoid(in_out);
#else
  // dense layer followed by a relu
  double_vector dense_out;
  dense(dense_out, in_out, dense1_w, dense1_b);
  relu(dense_out);

  // avgpool layer
  double_vector max_out;
  maxpool(max_out, dense_out, MAX_FILTER_SIZE, DENSE2_IN);

  // dense layer followed by a sigmoid
  dense(in_out, max_out, dense2_w, dense2_b);
  sigmoid(in_out);
#endif
}

//----------------------------------------------------------------

void Client::dense(double_vector &out, const double_vector &in,
                   const double_matrix &mat, const double_vector &bias) {
  plain_mat(out, in, mat, bias.size());

  for (size_t i = 0; i < out.size(); i++) out[i] += bias[i];
}

//----------------------------------------------------------------

void Client::conv1d(double_vector &out, const double_vector &in,
                    const std::vector<double_vector> &filter) {
  // 1d convolution
  // flattens output
  // same style padding
  size_t in_size = in.size();
  size_t filter_size = filter[0].size();
  size_t num_filter = filter.size();
  ssize_t filter_start = -(ssize_t)(filter_size - 1) / 2;

  out.resize(in_size * num_filter);

  for (size_t f = 0; f < num_filter; f++) {
    for (ssize_t i = 0; i < (ssize_t)in_size; i++) {
      double sum = 0.;
      for (ssize_t j = filter_start; j < filter_start + (ssize_t)filter_size;
           j++) {
        if (i + j >= (ssize_t)in_size || j + i < 0)  // padding with 0
          continue;
        sum += in[i + j] * filter[f][j - filter_start];
      }
      out[in_size * f + i] = sum;
    }
  }
}

//----------------------------------------------------------------

void Client::relu(double_vector &in_out) {
  for (auto &it : in_out) {
    if (it < 0.) it = 0.;
  }
}

//----------------------------------------------------------------

void Client::avgpool(double_vector &out, const double_vector &in,
                     uint64_t filter_size, uint64_t out_length) {
  // stride = 1
  out.resize(out_length);

  for (uint64_t i = 0; i < out_length; i++) {
    double sum = 0.;
    for (uint64_t j = 0; j < filter_size; j++) sum += in[i + j];
    out[i] = sum / filter_size;
  }
}

//----------------------------------------------------------------

void Client::maxpool(double_vector &out, const double_vector &in,
                     uint64_t filter_size, uint64_t out_length) {
  // stride = 1
  out.resize(out_length);

  for (uint64_t i = 0; i < out_length; i++) {
    double max = 0.;
    for (uint64_t j = 0; j < filter_size; j++)
      if (in[i + j] > max) max = in[i + j];
    out[i] = max;
  }
}

//----------------------------------------------------------------

void Client::plain_mat(double_vector &out, const double_vector &in,
                       const double_matrix &mat, size_t outsize) {
  if (out.size() != outsize) out.resize(outsize);

  for (uint64_t col = 0; col < outsize; col++) {
    double sum = 0;
    for (uint64_t row = 0; row < in.size(); row++) {
      sum += mat[row][col] * in[row];
    }
    out[col] = sum;
  }
}

//----------------------------------------------------------------

void Client::read_conv1d(std::ifstream &file, size_t num_filters,
                         size_t filter_size,
                         std::vector<double_vector> &out) const {
  if (!file.is_open()) throw std::runtime_error("Error opening weight file");
  std::string line;
  out.resize(num_filters, double_vector(filter_size));

  std::string expect_line = "Conv1d layer with " + std::to_string(num_filters) +
                            " filters of size " + std::to_string(filter_size) +
                            ":";
  std::getline(file, line);
  if (line != expect_line) throw ::std::runtime_error("Wrong file format");

  for (size_t i = 0; i < num_filters; i++) {
    expect_line = "Filter " + std::to_string(i);
    std::getline(file, line);
    if (line != expect_line) throw ::std::runtime_error("Wrong file format");
    for (size_t j = 0; j < filter_size; j++) {
      std::getline(file, line);
      out[i][j] = parse_double(line);
    }
  }
}

//----------------------------------------------------------------

void Client::read_dense(std::ifstream &file, size_t in_size, size_t out_size,
                        double_matrix &out_w, double_vector &out_b) const {
  if (!file.is_open()) throw std::runtime_error("Error opening weight file");
  std::string line;

  // pad with zero
  size_t mat_size = in_size;
  if (out_size > mat_size) mat_size = out_size;
  out_w.resize(mat_size, double_vector(mat_size, 0.));
  out_b.resize(out_size);

  std::string expect_line = "Dense layer of dimension (" +
                            std::to_string(in_size) + ", " +
                            std::to_string(out_size) + "):";
  std::getline(file, line);
  if (line != expect_line) throw ::std::runtime_error("Wrong file format");

  for (size_t i = 0; i < out_size; i++) {
    expect_line = "Columns " + std::to_string(i);
    std::getline(file, line);
    if (line != expect_line) throw ::std::runtime_error("Wrong file format");
    for (size_t j = 0; j < in_size; j++) {
      std::getline(file, line);
      out_w[j][i] = parse_double(line);
    }
  }

  expect_line = "Bias";
  std::getline(file, line);
  if (line != expect_line) throw ::std::runtime_error("Wrong file format");
  for (size_t i = 0; i < out_size; i++) {
    std::getline(file, line);
    out_b[i] = parse_double(line);
  }
}

//----------------------------------------------------------------

void Client::read_weights(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) throw std::runtime_error("Error opening weight file");

  // dense 1:
  read_dense(file, DENSE1_IN, DENSE1_OUT, dense1_w, dense1_b);
  // dense 2:
  read_dense(file, DENSE2_IN, OUT_SIZE, dense2_w, dense2_b);
}
