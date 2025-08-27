

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#ifndef cudaCpuDeviceId
#define cudaCpuDeviceId (-1)
#endif
#include "backprop.h"
#include "omp.h"

static double wall_time() {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

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

  double t_start = wall_time();
  
  int seed;

  // [old Rodinia Benchmark code]
  // if (argc!=2){
  // fprintf(stderr, "usage: backprop <num of input elements>\n");
  // exit(0);
  // }

  /* New Code Start */
  /* 2025-07-14 */
  if (argc < 2) {
      fprintf(stderr,
          "usage: ./backprop <in> [hid] [AB dev] [RM dev] [PL dev] [PF dev]\n");
      exit(1);
  }
  /* New Code End */

  layer_size = atoi(argv[1]);

  /* New Code Start */
  /* 2025-07-14 */
  hid_param = (argc >= 3) ? atol(argv[2]) : 2048;   // 缺省 2048

  /* ----------- 解析可选的 Unified Memory 优化 flag ----------- */
  extern int FLAG_AB, FLAG_RM, FLAG_PL, FLAG_PF;
  extern int DEV_AB, DEV_RM, DEV_PL, DEV_PF;

  /* 初始化默认值 */
  FLAG_AB = FLAG_RM = FLAG_PL = FLAG_PF = 0;
  DEV_AB = DEV_RM = DEV_PL = DEV_PF = -1;

  int idx = 3;   /* 从 argv[3] 开始 */
  while(idx < argc) {
      if (strcmp(argv[idx], "AB") == 0 && idx + 1 < argc) {
          FLAG_AB = 1;
          if (strcasecmp(argv[idx+1], "cpu") == 0)
              DEV_AB = cudaCpuDeviceId;
          else
              DEV_AB = atoi(argv[idx+1]);
          idx += 2;
      } else if (strcmp(argv[idx], "RM") == 0 && idx + 1 < argc) {
          FLAG_RM = 1;
          if (strcasecmp(argv[idx+1], "cpu") == 0)
              DEV_RM = cudaCpuDeviceId;
          else
              DEV_RM = atoi(argv[idx+1]);
          idx += 2;
      } else if (strcmp(argv[idx], "PL") == 0 && idx + 1 < argc) {
          FLAG_PL = 1;
          if (strcasecmp(argv[idx+1], "cpu") == 0)
              DEV_PL = cudaCpuDeviceId;
          else
              DEV_PL = atoi(argv[idx+1]);
          idx += 2;
      } else if (strcmp(argv[idx], "PF") == 0 && idx + 1 < argc) {
          FLAG_PF = 1;
          if (strcasecmp(argv[idx+1], "cpu") == 0)
              DEV_PF = cudaCpuDeviceId;
          else
              DEV_PF = atoi(argv[idx+1]);
          idx += 2;
      } else {
          fprintf(stderr, "Unknown or incomplete flag near '%s'\n", argv[idx]);
          exit(1);
      }
  }

  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  double t_end = wall_time();
  printf("Total elapsed time: %.3f seconds\n", t_end - t_start);

  exit(0);
}
