
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpicomm.h"

static int fwd_nbr=-1;
static int back_nbr=-1;
static int rank = -1;
static int size = -1;
int verbose = 0;
static int num_nodes;

void 
comm_init(int argc, char** argv)
{
  int gpu_per_node = 4;
  
  MPI_Init (&argc, &argv);
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#define MAX_PROCS 64
  int map[MAX_PROCS];

  if(size > MAX_PROCS){
    PRINTF("ERROR: # of mpi process (%d) exceed MAX_PROCS(%d)\n", size, MAX_PROCS);
    comm_exit(1);
  }
  //remapping
  num_nodes = size/gpu_per_node; //FIXME
  if (num_nodes == 0){
    num_nodes =1;
  }
  for(int i =0; i < size; i++){
    int j= i%num_nodes;
    int k =i/num_nodes;
    
    map[j*gpu_per_node+k] = i;
    
  }

  /*
  for(int i =0; i < size; i++){
    printf("virtual->real: map[%d]  ->%d\n", i, map[i]);
  }
  */

  int virtual_myself = -1;
  for (int i=0;i < size;i++){
    if (map[i] == rank){
      virtual_myself = i;
      break;
    }
  }

  if (virtual_myself < 0){
    printf("ERROR: virtual myself not found\n");
    exit(1);
  }
  
  back_nbr = map[(virtual_myself -1 +size)% size];
  fwd_nbr = map[(virtual_myself + 1)%size];
  
  printf("rank=%d, back_neighbor=%d, fwd_nbr=%d\n", 
	 rank, back_nbr, fwd_nbr);
  srand(rank*999);
  return;
}

int comm_gpuid()
{
  return rank/num_nodes;
}
int
comm_rank(void)
{
  return rank;
}

int
comm_size(void)
{
  return size;
}

unsigned long
comm_send(void* buf, int len, int dst)
{
  
  MPI_Request* request = (MPI_Request*)malloc(sizeof(MPI_Request));
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }

  int dstproc;
  int sendtag;
  if (dst == BACK_NBR){
    dstproc = back_nbr;
    sendtag = BACK_NBR;
  }else if (dst == FWD_NBR){
    dstproc = fwd_nbr;
    sendtag = FWD_NBR;
  }else{
    printf("ERROR: invalid dest\n");
    comm_exit(1);
  }

  MPI_Isend(buf, len, MPI_BYTE, dstproc, sendtag, MPI_COMM_WORLD, request);  
  return (unsigned long)request;  
}

unsigned long
comm_recv(void* buf, int len, int src)
{
  MPI_Request* request = (MPI_Request*)malloc(sizeof(MPI_Request));
  if (request == NULL){
    printf("ERROR: malloc failed for mpi request\n");
    comm_exit(1);
  }
  
  int srcproc;
  int recvtag; //recvtag is opposite to the sendtag
  if (src == BACK_NBR){
    srcproc = back_nbr;
    recvtag = FWD_NBR;
  }else if (src == FWD_NBR){
    srcproc = fwd_nbr;
    recvtag = BACK_NBR;
  }else{
    printf("ERROR: invalid source\n");
    comm_exit(1);
  }
  
  MPI_Irecv(buf, len, MPI_BYTE, srcproc, recvtag, MPI_COMM_WORLD, request);
  
  return (unsigned long)request;
}


//this request should be some return value from comm_recv
void 
comm_wait(unsigned long request)
{
  
  MPI_Status status;
  int rc = MPI_Wait( (MPI_Request*)request, &status);
  if (rc != MPI_SUCCESS){
    printf("ERROR: MPI_Wait failed\n");
    comm_exit(1);
  }
  
  free((void*)request);
  
  return;
}

//we always reduce one double value
void
comm_allreduce(double* data)
{
  double recvbuf;
  int rc = MPI_Allreduce ( data, &recvbuf,1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS){
    printf("ERROR: MPI_Allreduce failed\n");
    comm_exit(1);
  }
  
  *data = recvbuf;
  
  return;
} 
void
comm_barrier(void)
{
  MPI_Barrier(MPI_COMM_WORLD);  
}
void 
comm_cleanup()
{
  MPI_Finalize();
}

void
comm_exit(int ret)
{
  MPI_Finalize();
  exit(ret);
}

