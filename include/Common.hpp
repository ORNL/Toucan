#pragma once

#include <Stork_Core.hpp>
#include <nlohmann/json.hpp>

#include <mpi.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <utility>

#include <filesystem>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <utility>
#include <cstdint>
#include <climits>
#include <cfloat>
#include <thread>
#include <future>

namespace Toucan::Common{
    constexpr int self = 13;

    using std::vector;
    using std::string;
    using std::map;
    using std::tuple;
    using std::future;
    using std::promise;
    using std::thread;
    using std::ref;
    using std::move;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
}

// Needs to be declared outside of functions for cuda compilers (strange)
using std::max;
