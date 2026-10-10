#pragma once
#include "ple.h"
// Runs ple jobs split across both cores: core 0 (caller) does the first half,
// a pinned task on core 1 does the second half. Install with s->parallel = worker_parallel.
void worker_start(void);
void worker_parallel(ple_job_fn job, void *arg, int n);
