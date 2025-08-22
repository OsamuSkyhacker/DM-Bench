

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

static double wall_time() {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#include "backprop.h"
#include "omp.h"
/* ---------- global variables, visible to all translation units ---------- */
long hid_param = 2048;   /* 隐层大小，缺省 2048，需是 32 的倍数 */
/* ----------------------------------------------------------------------- */

extern char *strcpy();
extern void exit();

int layer_size = 0;

backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  // net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed) [old Rodinia Benchmark code]


  /* New Code Start */
  /* 2025-07-14 */
  /* 第二个命令行参数：直接指定隐层神经元个数 (32 的倍数) */
  extern long hid_param;   // 见 setup()

  long hid = hid_param;
  /* 对齐到 32，最少 32 */
  hid = (hid < 32 ? 32 : ( (hid + 31) & ~31 ));

  // /* 估算所需显存大小，仅供参考 */


  double bytes = 4.0 * ( 2.0*(layer_size+1)*(hid+1)
                     + (layer_size+1) + 2.0*(hid+1)
                     + ceil((double)hid/32.0)*hid );
  printf("Estimated memory usage: %.2f GiB\n", bytes / (1u<<30));

  net = bpnn_create(layer_size, (int)hid, 1);
  /* New Code End */
  
  printf("Input layer size : %d\n", layer_size);
  printf("Hidden layer size : %d\n", hid);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(argc, argv)
int argc;
char *argv[];
{
	int seed;
  double t_start = wall_time();

  // [old Rodinia Benchmark code]
  // if (argc!=2){
  // fprintf(stderr, "usage: backprop <num of input elements>\n");
  // exit(0);
  // }

  /* New Code Start */
  /* 2025-07-14 */
  if (argc < 2 || argc > 3) {
      fprintf(stderr,
          "usage: ./backprop <in> [hid (multiple of 32)]\n");
      exit(1);
  }
  /* New Code End */

  layer_size = atoi(argv[1]);

  /* New Code Start */
  /* 2025-07-14 */
  hid_param = (argc >= 3) ? atol(argv[2]) : 2048;   // 缺省 2048
  /* New Code End*/

  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();
  double t_end = wall_time();
  printf("Total elapsed time: %.3f s\n", t_end - t_start);
  exit(0);
}
