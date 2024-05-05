#ifndef RMPCPP_PLANNER_PARSER_H
#define RMPCPP_PLANNER_PARSER_H

#include <boost/program_options.hpp>

#include "rmpcpp_planner/testing/settings.h"
#include "rmpcpp_planner/testing/tester.h"

namespace po = boost::program_options;

class Parser {
 public:
  Parser(int argc, char* argv[]);

  bool parse();
  rmpcpp::TestSettings getSettings();
  rmpcpp::ParametersRMP getParameters();

 private:
  rmpcpp::ParametersRMP getRMPParameters();
  po::variables_map opts_;
};

#endif  // RMPCPP_PLANNER_PARSER_H
