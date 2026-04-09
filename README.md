# ML-Augmentation-for-SSSP-Algorithms

## Running Single Experiment

To plot a single run of an algorithm, look towards the runSearch() function in run.cpp.

There, you will see a line that looks like follows:

// pscase_type = ["randomD", "randomE", "randomG", "randomH", "randomT", "RD", "RF", "mix_real", "mix_gen", "mix_all"]
// pscase_predictor = ["false", "online", "offline", "blank"]
// frontier = ["bpq", "lapq"]
// countmin_predictor = ["false", "dedup", "ob", "npf", "hybrid"]
// countmin_type = ["false", "online", "offline", "blank"]
// BF_steps = int
spp_bmsspf::bmssp<distT> bmssp(adj, "randomT", "offline", "bpq", "ob", "online", 0);

This is the bmssp framework produced in this work, and there are 6 parameters that you may alter.

If you wish to run a different algorithm, change the first phrase preceded by "::". The most basic example includes:

spp_timed::bmssp<distT> bmssp(adj); -- This is the base version of bmssp.

To run a *bounded* bmssp, you must insert a threshold schedule. This can be seen below, where max_dist is the maximum distance fetched from a dijkstra run ahead of time.
spp_bounded_opt_k::bmssp(adj);
alg.set_threshold_schedule({max_dist + 1, oo});

To see how to run any of the algorithms in the "algs" directory, just look at the the namespace value, and replace "spp..." with it, as seen above.
The only exception is bmsspf, which requires parameters to be passed in at initilaisation.

The algorithm can then be executed by running "analysis.R". (Ensure that runSearch() is being called first).

run.cpp can also be compiled directly and ran itself, and you can manually change the graph family being used as input in 'run_class.json'.

## Running Batch Experiments

Batch experiments can be ran via analysis.R with the various functions that begin with "exp". Calling these in the main file will execute them.

## Plotting Results

As with running batch experiments, any of the plot functions can be ran, and they will plot any results in the "experiments" directory into the "figures" directory.