#include "worker.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

static SemaphoreHandle_t go, done;
static ple_job_fn cur_job; static void *cur_arg; static int cur_r0, cur_r1;

static void worker_task(void *p) {
    for (;;) {
        xSemaphoreTake(go, portMAX_DELAY);
        cur_job(cur_arg, cur_r0, cur_r1);
        xSemaphoreGive(done);
    }
}

void worker_start(void) {
    go = xSemaphoreCreateBinary(); done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker_task, "ple_w1", 4096, NULL, 5, NULL, 1);
}

void worker_parallel(ple_job_fn job, void *arg, int n) {
    int mid = n / 2;
    cur_job = job; cur_arg = arg; cur_r0 = mid; cur_r1 = n;
    xSemaphoreGive(go);                 // core 1 takes rows [mid, n)
    job(arg, 0, mid);                   // core 0 takes rows [0, mid)
    xSemaphoreTake(done, portMAX_DELAY);
}
