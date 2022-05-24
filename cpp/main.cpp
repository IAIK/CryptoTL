#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Client.h"
#include "Server.h"
#include "params.h"
#include "utils.h"

#define OUT_NAME "he_out.csv"

using namespace seal;

double cohens_kappa(size_t tp, size_t fp, size_t tn, size_t fn, size_t total);

int main(int argc, char* argv[]) {
  constexpr int CMP_PRECISION = 3;
  constexpr int MAX_RUNS = -1;
  constexpr bool VERBOSE = false;
  constexpr bool COMPARE_PLAIN = false;

  const char* server_weight_file;
  const char* input_file;
#if defined(SERVER_ONLY) || defined(STORE_INPUTS)
  if (argc == 3) {
    server_weight_file = argv[1];
    input_file = argv[2];
  } else {
    std::cout << "Usage: " << argv[0] << " <server_weight_file> <input_file>"
              << std::endl;
    return -1;
  }
#else
  const char* client_weight_file;
  if (argc == 4) {
    server_weight_file = argv[1];
    client_weight_file = argv[2];
    input_file = argv[3];
  } else {
    std::cout << "Usage: " << argv[0]
              << " <server_weight_file> <client_weight_file> <input_file>"
              << std::endl;
    return -1;
  }
#endif

  std::chrono::high_resolution_clock::time_point time_start, time_end;
  std::chrono::milliseconds time_diff;

  std::shared_ptr<seal::SEALContext> context;

  context = Client::create_CKKS_context(
      HE_TL_PARAM::MOD_DEGREE, HE_TL_PARAM::BIT_SCALE,
      HE_TL_PARAM::CKKS_FIX_SCALE, Server::required_levels());

  std::cout << "Generating keys..." << std::flush;
  Client client(context, HE_TL_PARAM::BIT_SCALE);
  auto gk_indices = Server::required_gk_indices(HE_TL_PARAM::USE_BSGS);
  client.create_gk(gk_indices);
  std::cout << "...done" << std::endl;
  client.print_parameters();

#if !defined(SERVER_ONLY) && !defined(STORE_INPUTS)
  std::cout << "Client reading weights..." << std::flush;
  client.read_weights(client_weight_file);
  std::cout << "...done" << std::endl;
#endif

  std::cout << "Client reading inputs..." << std::flush;
  client.read_inputs(input_file, true);
  std::cout << "...done" << std::endl;

  //----------------------------------------------------------------

  Server server(context, HE_TL_PARAM::BIT_SCALE);
  server.activate_bsgs(HE_TL_PARAM::USE_BSGS);
  server.set_gk(client.get_galois_keys());
  auto rlk = client.get_relin_keys();
  server.set_rk(rlk);

  std::cout << "Server reading weights..." << std::flush;
  server.read_weights(server_weight_file);
  std::cout << "...done" << std::endl;

  size_t outsize = server.get_out_size();
#if defined(STORE_INPUTS)
  std::ofstream storefile(OUT_NAME);
  if (!storefile.is_open())
    throw std::runtime_error("Error opening weight file");

  for (size_t i = 0; i < outsize; i++) {
    storefile << i << ",";
  }
  storefile << "label" << std::endl;
  storefile << std::setprecision(std::numeric_limits<double>::digits10 + 2);
#endif

  //----------------------------------------------------------------

  int64_t total_server_time = 0;
  int64_t total_client_time = 0;
  int64_t total_time = 0;

  bool correct = true;

#if !defined(STORE_INPUTS)
  size_t num_true_pos = 0;
  size_t num_false_pos = 0;
  size_t num_true_neg = 0;
  size_t num_false_neg = 0;
  size_t num_correct = 0;
  double loss = 0.;
#endif

  size_t runs = client.get_num_datasets();
  std::cout << "Running for " << runs << " datasets..." << std::endl;
  if (MAX_RUNS >= 0 && runs > MAX_RUNS) runs = MAX_RUNS;

  for (size_t run = 0; run < runs; run++) {
    if (VERBOSE) {
      std::cout << "-------------------------------------------------------"
                << std::endl;
      std::cout << "Run " << run + 1 << " of " << runs << std::endl;
      std::cout << "-------------------------------------------------------"
                << std::endl;
    }

    //----------------------------------------------------------------
    // Encrypt
    if (VERBOSE) {
      std::cout << "Encrypting..." << std::flush;
    }
    std::vector<double> input = client.get_dataset(run);
    int label = client.get_label(run);
    seal::Ciphertext ciph;
    client.encrypt(ciph, input);
    if (VERBOSE) {
      std::cout << "...done" << std::endl;
    }
    client.get_noise(ciph, true);

    //----------------------------------------------------------------
    // SERVER computation
    if (VERBOSE) {
      std::cout << "Server Computation..." << std::flush;
    }
    server.reset_level();
    time_start = std::chrono::high_resolution_clock::now();
    server.network(ciph);
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start);
    if (VERBOSE) {
      std::cout << "...done" << std::endl;
    }
    int64_t time = time_diff.count();
    total_server_time += time;
    total_time += time;
    if (VERBOSE) {
      std::cout << "Time: " << time << " milliseconds" << std::endl;
    }

    //----------------------------------------------------------------
    // decrypt
    client.get_noise(ciph, true);
    if (VERBOSE) {
      std::cout << "Decrypting..." << std::flush;
    }
    std::vector<double> output;
    client.decrypt(output, ciph);
    if (VERBOSE) {
      std::cout << "...done" << std::endl;
    }
    output.resize(outsize);

    //----------------------------------------------------------------
    // Write features
#if defined(STORE_INPUTS)
    if (VERBOSE) {
      std::cout << "Write output..." << std::flush;
    }
    for (size_t i = 0; i < outsize; i++) {
      storefile << output[i] << ",";
    }
    storefile << label << std::endl;
    if (VERBOSE) {
      std::cout << "...done" << std::endl;
    }
#endif

    if (COMPARE_PLAIN) {
      //----------------------------------------------------------------
      // Plain computation
      if (VERBOSE) {
        std::cout << "Doing compuation in plain..." << std::flush;
      }
      std::vector<double> output_p;
      time_start = std::chrono::high_resolution_clock::now();
      server.network_plain(output_p, input);
      time_end = std::chrono::high_resolution_clock::now();
      time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
          time_end - time_start);
      if (VERBOSE) {
        std::cout << "...done" << std::endl;
        std::cout << "Time: " << time_diff.count() << " milliseconds"
                  << std::endl;
      }

      //----------------------------------------------------------------
      // Compare Plain and HE
      output_p.resize(outsize);
      if (VERBOSE) {
        std::cout << "Comparing Plain and HE:" << std::endl;
      }
      if (compare_vectors(output_p, output, CMP_PRECISION)) {
        if (VERBOSE) {
          std::cout << "  Result is equal" << std::endl;
        }
      } else {
        correct = false;
        if (VERBOSE) {
          std::cout << "  ERROR: outputs mismatch" << std::endl;
          print_vector("Plain: ", output_p, CMP_PRECISION, std::cerr, outsize);
          print_vector("HE:    ", output, CMP_PRECISION, std::cerr, outsize);
        }
      }
      std::cout << std::endl;
    }

#if !defined(STORE_INPUTS)
    //----------------------------------------------------------------
    // Apply client network
    if (VERBOSE) {
      std::cout << "Applying Client network for HE output..." << std::flush;
    }
    time_start = std::chrono::high_resolution_clock::now();
    client.network(output);
    time_end = std::chrono::high_resolution_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start);
    time = time_diff.count();
    total_client_time += time;
    total_time += time;
    if (VERBOSE) {
      std::cout << "...done" << std::endl;
      std::cout << "Time: " << time << " milliseconds" << std::endl;
    }

    //----------------------------------------------------------------
    // Calculate Statistics
    if (VERBOSE) {
      std::cout << "Calculate statistics..." << std::flush;
    }
    int classified = 0;
    if (output[0] > 0.5) classified = 1;

    // stats copied from python impl
    double l = output[0];
    if (label == 0) l = 1. - l;
    l = -log((double)l);
    size_t acc = (classified == label) ? 1 : 0;
    size_t tp = (label == 1 && classified == label) ? 1 : 0;
    size_t fp = (label == 0 && classified != 0) ? 1 : 0;
    size_t tn = (label == 0 && classified == label) ? 1 : 0;
    size_t fn = (label == 1 && classified != 1) ? 1 : 0;
    num_true_pos += tp;
    num_false_pos += fp;
    num_true_neg += tn;
    num_false_neg += fn;
    num_correct += acc;
    loss += l;
    if (VERBOSE) {
      std::cout << "...done" << std::endl;
    }
#endif
  }

  std::cout << "-------------------------------------------------------"
            << std::endl;
  std::cout << "Final Result:" << std::endl;
  std::cout << "Average server time: " << total_server_time / runs
            << " milliseconds" << std::endl;
  std::cout << "Average client time: " << total_client_time / runs
            << " milliseconds" << std::endl;
  std::cout << "Average combined time: " << total_time / runs << " milliseconds"
            << std::endl;

#if !defined(STORE_INPUTS)
  double test_loss = loss / runs;
  double test_accuracy = (double)num_correct / runs;
  double recall = (double)num_true_pos / (num_true_pos + num_false_neg);
  double precision = (double)num_true_pos / (num_true_pos + num_false_pos);
  double f1_score = 2 * precision * recall / (precision + recall);
  double kappa_score = cohens_kappa(num_true_pos, num_false_pos, num_true_neg,
                                    num_false_neg, runs);
  print_double("Test Loss:", test_loss, 3, std::cout);
  print_double("Test Accuracy:", test_accuracy, 3, std::cout);
  print_double("Test F1 Score (Macro):", f1_score, 3, std::cout);
  print_double("Test Cohen's Kappa Score:", kappa_score, 3, std::cout);
#else
  (void)&print_double;  // silence warning
#endif

  if (!correct) {
    std::cout << "There was at least one error..." << std::endl;
    return -1;
  }

  return 0;
}

//----------------------------------------------------------------

double cohens_kappa(size_t tp, size_t fp, size_t tn, size_t fn, size_t total) {
  double prob_tp_fp = (double)(tp + fp) / total;
  double prob_tp_fn = (double)(tp + fn) / total;
  double prob_tn_fn = (double)(tn + fn) / total;
  double prob_tn_fp = (double)(tn + fp) / total;

  double sum_prob_pos = prob_tp_fp * prob_tp_fn;
  double sum_prob_neg = prob_tn_fn * prob_tn_fp;
  double chance_agree = sum_prob_pos + sum_prob_neg;
  double agree = (double)(tp + tn) / total;
  return (agree - chance_agree) / (1 - chance_agree);
}
