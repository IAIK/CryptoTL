#include "Server.h"

#include <assert.h>

#include <fstream>
#include <iostream>

using namespace seal;

//----------------------------------------------------------------

Server::Server(std::shared_ptr<seal::SEALContext> con, size_t sc)
    : context(con),
      evaluator(*context),
      plain_mod(0),
      scale(pow(2.0, sc)),
      bitscale(sc),
      current_level(0) {
  ckks_encoder = std::make_unique<seal::CKKSEncoder>(*context);
  slots = ckks_encoder->slot_count();
}

//----------------------------------------------------------------

void Server::reset_level() { current_level = 0; }

//----------------------------------------------------------------

uint64_t Server::get_level() const { return current_level; }

//----------------------------------------------------------------

void Server::set_gk(seal::GaloisKeys &galois) {
  galois_keys = galois;
  gk_set = true;
}

//----------------------------------------------------------------

void Server::set_rk(seal::RelinKeys &relin) {
  relin_keys = relin;
  relin_set = true;
}

//----------------------------------------------------------------

void Server::CKKS_diagonal(Ciphertext &in_out, const double_matrix &mat) {
  size_t matrix_dim = mat.size();

  if (matrix_dim != slots && matrix_dim * 2 > slots)
    throw std::runtime_error("too little slots for matmul implementation!");

  if (slots != matrix_dim) {
    Ciphertext in_out_rot;
    evaluator.rotate_vector(in_out, -((int)matrix_dim), galois_keys,
                            in_out_rot);
    evaluator.add_inplace(in_out, in_out_rot);
  }

  // diagonal method preperation:
  std::vector<Plaintext> matrix;
  matrix.reserve(matrix_dim);
  for (auto i = 0ULL; i < matrix_dim; i++) {
    std::vector<double> diag;
    diag.reserve(matrix_dim);
    for (auto j = 0ULL; j < matrix_dim; j++) {
      diag.push_back(mat[(i + j) % matrix_dim][j]);
    }
    Plaintext row;
    ckks_encoder->encode(diag, scale, row);
    if (current_level != 0)
      evaluator.mod_switch_to_inplace(row, in_out.parms_id());
    matrix.push_back(row);
  }

  Ciphertext sum = in_out;
  evaluator.multiply_plain_inplace(sum, matrix[0]);
  for (auto i = 1ULL; i < matrix_dim; i++) {
    Ciphertext tmp;
    evaluator.rotate_vector_inplace(in_out, 1, galois_keys);
    try {
      evaluator.multiply_plain(in_out, matrix[i], tmp);
      evaluator.add_inplace(sum, tmp);
    } catch (std::logic_error &ex) {
      // ignore transparent ciphertext
    }
  }
  in_out = sum;
}

//----------------------------------------------------------------

void Server::CKKS_babystep_giantstep(Ciphertext &in_out,
                                     const double_matrix &mat) {
  size_t matrix_dim = mat.size();

  if (matrix_dim != slots && matrix_dim * 2 > slots)
    throw std::runtime_error("too little slots for matmul implementation!");

  if (bsgs_n1 * bsgs_n2 != matrix_dim)
    throw std::runtime_error("wrong bsgs parameters");

  // baby step giant step method preperation:
  std::vector<Plaintext> matrix;
  matrix.reserve(matrix_dim);
  for (auto i = 0ULL; i < matrix_dim; i++) {
    std::vector<double> diag;
    auto k = i / bsgs_n1;
    diag.reserve(matrix_dim + k * bsgs_n1);

    for (auto j = 0ULL; j < matrix_dim; j++) {
      diag.push_back(mat[(i + j) % matrix_dim][j]);
    }
    // rotate:
    if (k)
      std::rotate(diag.begin(), diag.begin() + diag.size() - k * bsgs_n1,
                  diag.end());

    // prepare for non-full-packed rotations
    if (slots != matrix_dim) {
      for (uint64_t index = 0; index < k * bsgs_n1; index++) {
        diag.push_back(diag[index]);
        diag[index] = 0;
      }
    }

    Plaintext row;
    ckks_encoder->encode(diag, scale, row);
    if (current_level != 0)
      evaluator.mod_switch_to_inplace(row, in_out.parms_id());
    matrix.push_back(row);
  }

  // prepare for non-full-packed rotations
  if (slots != matrix_dim) {
    Ciphertext in_out_rot;
    evaluator.rotate_vector(in_out, -((int)matrix_dim), galois_keys,
                            in_out_rot);
    evaluator.add_inplace(in_out, in_out_rot);
  }

  Ciphertext temp;
  Ciphertext outer_sum;
  Ciphertext inner_sum;

  // prepare rotations
  std::vector<Ciphertext> rot;
  rot.resize(bsgs_n1);
  rot[0] = in_out;
  for (uint64_t j = 1; j < bsgs_n1; j++)
    evaluator.rotate_vector(rot[j - 1], 1, galois_keys, rot[j]);

  for (uint64_t k = 0; k < bsgs_n2; k++) {
    evaluator.multiply_plain(rot[0], matrix[k * bsgs_n1], inner_sum);
    for (uint64_t j = 1; j < bsgs_n1; j++) {
      try {
        evaluator.multiply_plain(rot[j], matrix[k * bsgs_n1 + j], temp);
        evaluator.add_inplace(inner_sum, temp);
      } catch (std::logic_error &ex) {
        // ignore transparent ciphertext
      }
    }
    if (!k)
      outer_sum = inner_sum;
    else {
      evaluator.rotate_vector_inplace(inner_sum, k * bsgs_n1, galois_keys);
      evaluator.add_inplace(outer_sum, inner_sum);
    }
  }
  in_out = outer_sum;
}

//----------------------------------------------------------------

void Server::activate_bsgs(bool bsgs) { use_bsgs = bsgs; }

//----------------------------------------------------------------

void Server::cipher_from_file(seal::Ciphertext &ciph) {
  std::ifstream c;
  c.open(RESULT_FILE);
  ciph.load(*context, c);
}

//----------------------------------------------------------------

std::shared_ptr<seal::SEALContext> Server::context_from_file() {
  EncryptionParameters parms;

  std::ifstream param;
  param.open(PARAM_FILE);
  parms.load(param);

  sec_level_type sec = sec_level_type::tc128;
  return std::make_shared<seal::SEALContext>(parms, true, sec);
}

//----------------------------------------------------------------

uint64_t Server::keys_from_file(bool relin) {
  std::ifstream gk;
  gk.open(GK_FILE);
  galois_keys.load(*context, gk);
  gk_set = true;

  if (relin) {
    std::ifstream rk;
    rk.open(RK_FILE);
    relin_keys.load(*context, rk);
    relin_set = true;
  }
  return 0;
}

//----------------------------------------------------------------

void Server::print_parameters() {
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

void Server::cipher_to_file(seal::Ciphertext &ciph) {
  std::ofstream c_out;
  c_out.open(RESULT_FILE);
  ciph.save(c_out);
}

//----------------------------------------------------------------

void Server::plain_mat(double_vector &out, const double_vector &in,
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
// NN code:
//----------------------------------------------------------------

void Server::network(seal::Ciphertext &in_out) {
  if (!gk_set || !relin_set)
    throw std::runtime_error("Galois keys or relin keys not set.");

  // conv layer
  conv1d(in_out, IN_SIZE, conv1d_filter);

  // dense layer followed by a relu
  bsgs_n1 = DENSE1_BSGS_N1;
  bsgs_n2 = DENSE1_BSGS_N2;
  dense(in_out, dense1_w, dense1_b);
  relu3(in_out);

  // avgpool layer
  avgpool(in_out, AVG_FILTER_SIZE, DENSE2_IN);

  // dense layer
  bsgs_n1 = DENSE2_BSGS_N1;
  bsgs_n2 = DENSE2_BSGS_N2;
  dense(in_out, dense2_w, dense2_b, true);
}

//----------------------------------------------------------------

void Server::dense(seal::Ciphertext &in_out, const double_matrix &mat,
                   const double_vector &bias, bool skip_rescale) {
  Plaintext bias_enc;

  if (use_bsgs)
    CKKS_babystep_giantstep(in_out, mat);
  else
    CKKS_diagonal(in_out, mat);

  if (!skip_rescale) evaluator.rescale_to_next_inplace(in_out);
  ckks_encoder->encode(bias, in_out.scale(), bias_enc);
  evaluator.mod_switch_to_inplace(bias_enc, in_out.parms_id());

  evaluator.add_plain_inplace(in_out, bias_enc);
  current_level++;
}

//----------------------------------------------------------------

void Server::conv1d(seal::Ciphertext &in_out, size_t in_size,
                    const std::vector<double_vector> &filter) {
  // 1d convolution
  // flattens output
  // same style padding
  size_t filter_size = filter[0].size();
  size_t num_filter = filter.size();
  ssize_t filter_start = -(ssize_t)(filter_size - 1) / 2;

  // prepare in_out for multiple filter (flattened output)
  Ciphertext in_flat = in_out;
  for (size_t i = 0; i < num_filter - 1; i++) {
    // todo: less rotations?
    evaluator.rotate_vector_inplace(in_flat, -in_size, galois_keys);
    evaluator.add_inplace(in_flat, in_out);
  }

  Plaintext fil_encoded;
  Ciphertext mul;

  for (ssize_t j = filter_start; j < filter_start + (ssize_t)filter_size; j++) {
    double_vector fil(in_size * num_filter);
    for (size_t f = 0; f < num_filter; f++) {
      for (size_t i = 0; i < in_size; i++) {
        ssize_t bound = in_size - 1 - i + j;
        if (bound >= (ssize_t)in_size || bound < 0)
          fil[f * in_size + i] = 0.;
        else
          fil[f * in_size + i] = filter[f][j - filter_start];
      }
    }
    ckks_encoder->encode(fil, scale, fil_encoded);
    if (current_level != 0)
      evaluator.mod_switch_to_inplace(fil_encoded, in_out.parms_id());
    evaluator.multiply_plain(in_flat, fil_encoded, mul);

    // rotate and add
    if (j == 0 && j == filter_start)
      in_out = mul;
    else if (j == 0)
      evaluator.add_inplace(in_out, mul);
    else {
      evaluator.rotate_vector_inplace(mul, j, galois_keys);
      if (j == filter_start)
        in_out = mul;
      else
        evaluator.add_inplace(in_out, mul);
    }
  }

  evaluator.rescale_to_next_inplace(in_out);

  current_level += 1;
}

//----------------------------------------------------------------

void Server::conv2d(seal::Ciphertext &in_out, size_t in_size,
                    const std::vector<double_matrix> &kernel) {
  //   2D convolution
  // flattens output
  // same style padding
  size_t num_filter = kernel.size();
  size_t kernel_rows = kernel[0].size();
  size_t kernel_cols = kernel[0][0].size();
  // only implement this
  if (kernel_rows != kernel_cols) {
    throw std::runtime_error("Only square kernels allowed");
  }
  if (kernel_rows % 2 != 1) {
    throw std::runtime_error("Only odd kernels allowed");
  }

  Plaintext fil_encoded;
  Ciphertext mul;
  Ciphertext in = in_out;

  ssize_t filter_start = -(ssize_t)(kernel_rows - 1) / 2;
  ssize_t filters_start = -(ssize_t)(kernel_rows * kernel_cols - 1) / 2;
  size_t in_size_square = in_size * in_size;

  for (ssize_t fi = filter_start; fi < filter_start + (ssize_t)kernel_rows;
       fi++) {
    for (ssize_t fj = filter_start; fj < filter_start + (ssize_t)kernel_cols;
         fj++) {
      double_vector fil(in_size * in_size, 0.);
      for (size_t f = 0; f < num_filter; f++) {
        for (ssize_t i = 0; i < (ssize_t)in_size; i++) {
          for (ssize_t j = 0; j < (ssize_t)in_size; j++) {
            ssize_t bound_x = in_size - 1 - i - fi;
            ssize_t bound_y = in_size - 1 - j - fj;

            if (bound_x >= (ssize_t)in_size || bound_x < 0 ||
                bound_y >= (ssize_t)in_size || bound_y < 0) {
              continue;
            }
            fil[f * in_size_square + i * in_size + j] =
                kernel[f][kernel_rows - 1 - fi + filter_start]
                      [kernel_rows - 1 - fj + filter_start];
          }
        }
      }

      ckks_encoder->encode(fil, scale, fil_encoded);
      if (current_level != 0)
        evaluator.mod_switch_to_inplace(fil_encoded, in_out.parms_id());
      evaluator.multiply_plain(in, fil_encoded, mul);

      // rotate and add
      ssize_t coord = fi * kernel_rows + fj;
      ssize_t rot = fi * in_size + fj;
      if (coord == 0 && coord == filters_start) {
        in_out = mul;
      } else if (coord == 0) {
        evaluator.add_inplace(in_out, mul);
      } else {
        evaluator.rotate_vector_inplace(mul, -rot, galois_keys);
        if (coord == filters_start)
          in_out = mul;
        else
          evaluator.add_inplace(in_out, mul);
      }
    }
  }

  evaluator.rescale_to_next_inplace(in_out);

  current_level += 1;
}

//----------------------------------------------------------------

// void Server::relu2(seal::Ciphertext &in_out) {
//   constexpr double x0 = 0.;
//   constexpr double x1 = 0.;
//   constexpr double x2 = 0.;

//   Ciphertext x2_enc, x1_enc;
//   Plaintext c0, c1, c2;

//   double_vector vec2(slots, x2);
//   ckks_encoder->encode(vec2, in_out.scale(), c2);

//   evaluator.square(in_out, x2_enc);
//   evaluator.relinearize_inplace(x2_enc, relin_keys);

//   evaluator.rescale_to_next_inplace(x2_enc);
//   evaluator.mod_switch_to_inplace(c2, x2_enc.parms_id());
//   double_vector vec1(slots, x1);
//   ckks_encoder->encode(vec1, x2_enc.scale(), c1);
//   evaluator.mod_switch_to_inplace(in_out, x2_enc.parms_id());
//   evaluator.mod_switch_to_inplace(c1, in_out.parms_id());

//   evaluator.multiply_plain_inplace(in_out, c1);
//   evaluator.multiply_plain_inplace(x2_enc, c2);

//   evaluator.rescale_to_next_inplace(x2_enc);
//   evaluator.rescale_to_next_inplace(in_out);
//   double_vector vec0(slots, x0);
//   ckks_encoder->encode(vec0, x2_enc.scale(), c0);
//   evaluator.mod_switch_to_inplace(c0, x2_enc.parms_id());

//   evaluator.add_plain_inplace(in_out, c0);
//   evaluator.add_inplace(in_out, x2_enc);

//   current_level += current_level + 2;
// }

//----------------------------------------------------------------

void Server::relu3(seal::Ciphertext &in_out) {
  constexpr double x0 = 0.49383;
  constexpr double x1 = 0.59259;
  constexpr double x2 = 0.092593;
  constexpr double x3 = -0.0061728;

  Ciphertext x2_enc, x3_enc, x1_s;
  Plaintext c0, c1, c2, c3;

  double_vector vec3(slots, x3);
  ckks_encoder->encode(vec3, in_out.scale(), c3);
  evaluator.mod_switch_to_inplace(c3, in_out.parms_id());

  evaluator.square(in_out, x2_enc);
  evaluator.relinearize_inplace(x2_enc, relin_keys);

  evaluator.multiply_plain(in_out, c3, x1_s);

  evaluator.rescale_to_next_inplace(x2_enc);
  evaluator.rescale_to_next_inplace(x1_s);

  evaluator.multiply(x1_s, x2_enc, x3_enc);
  evaluator.relinearize_inplace(x3_enc, relin_keys);

  evaluator.rescale_to_next_inplace(x3_enc);

  double_vector vec2(slots, x2);
  ckks_encoder->encode(vec2, x1_s.scale(), c2);
  evaluator.mod_switch_to_inplace(c2, x2_enc.parms_id());

  evaluator.multiply_plain_inplace(x2_enc, c2);

  if (NORMALIZE_SET_SCALE) {
    // mult with constant and set scale
    double_vector vec1(slots, x1);
    ckks_encoder->encode(vec1, in_out.scale(), c1);
    if (current_level != 0)
      evaluator.mod_switch_to_inplace(c1, in_out.parms_id());
    evaluator.multiply_plain_inplace(in_out, c1);
    evaluator.rescale_to_next_inplace(in_out);
    in_out.scale() = x3_enc.scale();
  } else {
    // mult with 1 first
    double_vector vec(slots, 1.0);
    ckks_encoder->encode(vec, in_out.scale(), c1);
    if (current_level != 0)
      evaluator.mod_switch_to_inplace(c1, in_out.parms_id());
    evaluator.multiply_plain_inplace(in_out, c1);
    evaluator.rescale_to_next_inplace(in_out);
    // mult with constant
    double_vector vec1(slots, x1);
    ckks_encoder->encode(vec1, x1_s.scale(), c1);
    evaluator.mod_switch_to_inplace(c1, in_out.parms_id());
    evaluator.multiply_plain_inplace(in_out, c1);
    evaluator.rescale_to_next_inplace(in_out);
  }

  parms_id_type last_parms_id = x3_enc.parms_id();

  double_vector vec0(slots, x0);
  ckks_encoder->encode(vec0, x3_enc.scale(), c0);
  evaluator.mod_switch_to_inplace(c0, last_parms_id);
  evaluator.mod_switch_to_inplace(in_out, last_parms_id);
  evaluator.rescale_to_next_inplace(x2_enc);
  evaluator.mod_switch_to_inplace(x2_enc, last_parms_id);

  evaluator.add_plain_inplace(in_out, c0);
  evaluator.add_inplace(in_out, x2_enc);
  evaluator.add_inplace(in_out, x3_enc);

  current_level += current_level + 2;
}

//----------------------------------------------------------------

// void Server::relu4(seal::Ciphertext &in_out) {
//   // return 0.69136 + 0.49455 * in + 0.083061 * in^2 - 0.00083975 * in^3
//   - 0.00037445 * in^4
//   constexpr double x0 = 0.69136;
//   constexpr double x1 = 0.49455;
//   constexpr double x2 = 0.083061;
//   constexpr double x3 = -0.00083975;
//   constexpr double x4 = -0.00037445;

//   Ciphertext x2_enc, x3_enc, x4_enc;
//   Plaintext c0, c1, c2, c3, c4;

//   double_vector vec0(slots, x0);
//   double_vector vec1(slots, x1);
//   double_vector vec2(slots, x2);
//   double_vector vec3(slots, x3);
//   double_vector vec4(slots, x4);

//   auto scale1 = in_out.scale();
//   ckks_encoder->encode(vec4, scale1, c4);

//   evaluator.square(in_out, x2_enc);
//   evaluator.relinearize_inplace(x2_enc, relin_keys);
//   evaluator.rescale_to_next_inplace(x2_enc);

//   evaluator.square(x2_enc, x4_enc);
//   evaluator.relinearize_inplace(x4_enc, relin_keys);
//   evaluator.rescale_to_next_inplace(x4_enc);

//   ckks_encoder->encode(vec1, x4_enc.scale(), c1);

//   if (x3 != 0.) {
//     ckks_encoder->encode(vec3, scale1, c3);
//     auto tmp = in_out;
//     evaluator.mod_switch_to_inplace(c3, tmp.parms_id());
//     evaluator.multiply_plain_inplace(tmp, c3);
//     evaluator.rescale_to_next_inplace(tmp);

//     evaluator.multiply(x2_enc, tmp, x3_enc);
//     evaluator.relinearize_inplace(x3_enc, relin_keys);
//     evaluator.rescale_to_next_inplace(x3_enc);
//   }

//   evaluator.mod_switch_to_inplace(in_out, x4_enc.parms_id());

//   evaluator.mod_switch_to_inplace(c4, x4_enc.parms_id());
//   evaluator.multiply_plain_inplace(x4_enc, c4);
//   evaluator.rescale_to_next_inplace(x4_enc);

//   ckks_encoder->encode(vec2, x2_enc.scale(), c2);

//   evaluator.mod_switch_to_inplace(c2, x2_enc.parms_id());
//   evaluator.multiply_plain_inplace(x2_enc, c2);
//   evaluator.rescale_to_next_inplace(x2_enc);

//   evaluator.mod_switch_to_inplace(c1, in_out.parms_id());
//   evaluator.multiply_plain_inplace(in_out, c1);
//   evaluator.rescale_to_next_inplace(in_out);

//   if (x0 != 0.) {
//     ckks_encoder->encode(vec0, in_out.scale(), c0);
//     evaluator.mod_switch_to_inplace(c0, in_out.parms_id());
//     evaluator.add_plain_inplace(in_out, c0);
//   }

//   if (NORMALIZE_SET_SCALE) {
//     evaluator.mod_switch_to_inplace(x2_enc, in_out.parms_id());
//     x2_enc.scale() = x4_enc.scale();
//     if (x3 != 0.) {
//       evaluator.mod_switch_to_inplace(x3_enc, in_out.parms_id());
//       x3_enc.scale() = x4_enc.scale();
//     }
//   } else {
//     // mult with 1 first
//     double_vector vec(slots, 1.0);
//     ckks_encoder->encode(vec, scale1, c1);

//     evaluator.mod_switch_to_inplace(c1, x2_enc.parms_id());
//     evaluator.multiply_plain_inplace(x2_enc, c1);
//     evaluator.rescale_to_next_inplace(x2_enc);
//     if (x3 != 0.) {
//       evaluator.multiply_plain_inplace(x3_enc, c1);
//       evaluator.rescale_to_next_inplace(x3_enc);
//     }
//   }
//   evaluator.add_inplace(in_out, x2_enc);
//   if (x3 != 0.) evaluator.add_inplace(in_out, x3_enc);
//   evaluator.add_inplace(in_out, x4_enc);

//   current_level += current_level + 3;
// }

//----------------------------------------------------------------

void Server::avgpool(seal::Ciphertext &in_out, uint64_t filter_size,
                     uint64_t out_length) {
  // stride = 1
  Plaintext mask_enc;
  Ciphertext rotated;

  double factor = 1. / filter_size;
  double_vector mask(out_length, factor);

  ckks_encoder->encode(mask, scale, mask_enc);
  if (current_level != 0)
    evaluator.mod_switch_to_inplace(mask_enc, in_out.parms_id());

  rotated = in_out;
  for (uint64_t i = 0; i < filter_size - 1; i++) {
    evaluator.rotate_vector(rotated, 1, galois_keys, rotated);
    evaluator.add_inplace(in_out, rotated);
  }

  evaluator.multiply_plain_inplace(in_out, mask_enc);
  evaluator.rescale_to_next_inplace(in_out);

  current_level += 1;
}

//----------------------------------------------------------------
// NN code plain:
//----------------------------------------------------------------

void Server::network_plain(double_vector &out, const double_vector &in) {
  // conv layer
  double_vector conv_out;
  conv1d_plain(conv_out, in, conv1d_filter);

  // dense layer followed by a relu
  double_vector dense_out;
  dense_plain(dense_out, conv_out, dense1_w, dense1_b);
  relu3_plain(dense_out);

  // avgpool layer
  double_vector avg_out;
  avgpool_plain(avg_out, dense_out, AVG_FILTER_SIZE, DENSE2_IN);

  // dense layer
  dense_plain(out, avg_out, dense2_w, dense2_b);
}

//----------------------------------------------------------------

void Server::dense_plain(double_vector &out, const double_vector &in,
                         const double_matrix &mat, const double_vector &bias) {
  plain_mat(out, in, mat, bias.size());

  for (size_t i = 0; i < out.size(); i++) out[i] += bias[i];
}

//----------------------------------------------------------------

void Server::conv1d_plain(double_vector &out, const double_vector &in,
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

void Server::conv2d_plain(double_matrix &out, const double_matrix &in,
                          const double_matrix &kernel) {
  //   1x 2D convolution
  //   same style padding
  size_t kernel_rows = kernel.size();
  size_t kernel_cols = kernel[0].size();

  size_t in_rows = in.size();
  size_t in_cols = in.size();
  // only implement this
  if (kernel_rows != kernel_cols) {
    throw std::runtime_error("Only square kernels allowed");
  }
  if (in_rows != in_cols) {
    throw std::runtime_error("Only square inputs allowed");
  }
  if (kernel_rows % 2 != 1) {
    throw std::runtime_error("Only odd kernels allowed");
  }

  ssize_t filter_start = -(ssize_t)(kernel_rows - 1) / 2;

  out.clear();
  out.resize(in_rows, std::vector<double>(in_cols, 0.));

  for (ssize_t i = 0; i < (ssize_t)in_rows; i++) {
    for (ssize_t j = 0; j < (ssize_t)in_cols; j++) {
      double sum = 0.;
      for (ssize_t fi = filter_start; fi < filter_start + (ssize_t)kernel_rows;
           fi++) {
        for (ssize_t fj = filter_start;
             fj < filter_start + (ssize_t)kernel_cols; fj++) {
          ssize_t x = i + fi;
          ssize_t y = j + fj;

          if (x < 0 || y < 0 || x >= (ssize_t)in_rows ||
              y >= (ssize_t)in_cols) {
            // padding with 0
            continue;
          }

          sum += in[x][y] * kernel[fi - filter_start][fj - filter_start];
        }
      }
      out[i][j] = sum;
    }
  }
}

//----------------------------------------------------------------

// void Server::relu2_plain(double_vector &in_out) {
//   constexpr double x0 = 0.;
//   constexpr double x1 = 0.;
//   constexpr double x2 = 0.;

//   for (auto &it : in_out) {
//     double it2 = it * it;
//     it = x0 + x1 * it + x2 * it2;
//   }
// }

//----------------------------------------------------------------

void Server::relu3_plain(double_vector &in_out) {
  constexpr double x0 = 0.49383;
  constexpr double x1 = 0.59259;
  constexpr double x2 = 0.092593;
  constexpr double x3 = -0.0061728;

  for (auto &it : in_out) {
    double it2 = it * it;
    it = x0 + x1 * it + x2 * it2 + x3 * it * it2;
  }
}

//----------------------------------------------------------------

// void Server::relu4_plain(double_vector &in_out) {
//   // return 0.69136 + 0.49455 * in + 0.083061 * in^2 - 0.00083975 * in^3
//   - 0.00037445 * in^4
//   constexpr double x0 = 0.69136;
//   constexpr double x1 = 0.49455;
//   constexpr double x2 = 0.083061;
//   constexpr double x3 = -0.00083975;
//   constexpr double x4 = -0.00037445;

//   for (auto &it : in_out) {
//     double it2 = it * it;
//     it = x0 + x1 * it + x2 * it2 + x3 * it * it2 + x4 * it2 * it2;
//   }
// }

//----------------------------------------------------------------

void Server::avgpool_plain(double_vector &out, const double_vector &in,
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

double Server::parse(std::string in) {
  double out;
  std::stringstream buffer(in);
  buffer >> out;
  if (buffer.bad() || !buffer.eof())
    throw std::runtime_error("Error parsing string to double");
  return out;
}

//----------------------------------------------------------------

void Server::read_conv1d(std::ifstream &file, size_t num_filters,
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
      out[i][j] = parse(line);
    }
  }
}

//----------------------------------------------------------------

void Server::read_dense(std::ifstream &file, size_t in_size, size_t out_size,
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
      out_w[j][i] = parse(line);
    }
  }

  expect_line = "Bias";
  std::getline(file, line);
  if (line != expect_line) throw ::std::runtime_error("Wrong file format");
  for (size_t i = 0; i < out_size; i++) {
    std::getline(file, line);
    out_b[i] = parse(line);
  }
}

//----------------------------------------------------------------

void Server::read_weights(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) throw std::runtime_error("Error opening weight file");

  // conv1d:
  read_conv1d(file, CONV1_NUM_FILTERS, CONV1_FILTER_SIZE, conv1d_filter);
  // dense 1:
  read_dense(file, DENSE1_IN, DENSE1_OUT, dense1_w, dense1_b);
  // dense 2:
  read_dense(file, DENSE2_IN, OUT_SIZE, dense2_w, dense2_b);
}

//--------------------------------------------------------

std::vector<int> Server::required_gk_indices(bool use_bsgs) {
  ssize_t filter_start = -(ssize_t)(CONV1_FILTER_SIZE - 1) / 2;

  std::vector<int> gks;
  gks.push_back(1);
  gks.push_back(-((int)IN_SIZE));
  gks.push_back(-(int)(DENSE1_BSGS_N1 * DENSE1_BSGS_N2));
  gks.push_back(-(int)(DENSE2_BSGS_N1 * DENSE2_BSGS_N2));

  for (ssize_t i = filter_start; i < filter_start + (ssize_t)CONV1_FILTER_SIZE;
       i++)
    gks.push_back(i);
  if (use_bsgs) {
    for (uint64_t l = 1; l < DENSE1_BSGS_N2; l++)
      gks.push_back(l * DENSE1_BSGS_N1);
    for (uint64_t l = 1; l < DENSE2_BSGS_N2; l++)
      gks.push_back(l * DENSE2_BSGS_N1);
  }
  return gks;
}

//--------------------------------------------------------

uint64_t Server::required_levels() { return LEVELS; }

//--------------------------------------------------------

uint64_t Server::get_out_size() { return OUT_SIZE; }
