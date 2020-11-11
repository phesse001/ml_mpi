/*
Copyright (c) 2016-2020 Jeremy Iverson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* assert */
#include <assert.h>

/* fabs */
#include <math.h>

/* MPI API */
#include <mpi.h>

/* printf, fopen, fclose, fscanf, scanf */
#include <stdio.h>

/* EXIT_SUCCESS, malloc, calloc, free, qsort */
#include <stdlib.h>

#define MPI_SIZE_T MPI_UNSIGNED_LONG

struct distance_metric {
  size_t viewer_id;
  double distance;
};

static int
cmp(void const *ap, void const *bp)
{
  struct distance_metric const a = *(struct distance_metric*)ap;
  struct distance_metric const b = *(struct distance_metric*)bp;

  return a.distance < b.distance ? -1 : 1;
}

int
main(int argc, char * argv[])
{
  int ret, p, rank;
  size_t n, m, k;
  double * rating;

  /* Initialize MPI environment. */
  ret = MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS == ret);

  /* Get size of world communicator. */
  ret = MPI_Comm_size(MPI_COMM_WORLD, &p);
  assert(ret == MPI_SUCCESS);

  /* Get my rank. */
  ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(ret == MPI_SUCCESS);

  /*try to create distance type :(*/
  const int nitems=2;
  int          blocklengths[2] = {1,1};
  MPI_Datatype types[2] = {MPI_SIZE_T, MPI_DOUBLE};
  MPI_Datatype MPI_DISTANCE;
  MPI_Aint     offsets[2];

  offsets[0] = offsetof(struct distance_metric, viewer_id);
  offsets[1] = offsetof(struct distance_metric, distance);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_DISTANCE);
  MPI_Type_commit(&MPI_DISTANCE);

  /* Validate command line arguments. */
  assert(2 == argc);

  /* Read input --- only if your rank 0. */
  if (0 == rank) {
    /* ... */
    char const * const fn = argv[1];

    /* Validate input. */
    assert(fn);

    /* Open file. */
    FILE * const fp = fopen(fn, "r");
    assert(fp);

    /* Read file. */
    fscanf(fp, "%zu %zu", &n, &m);

    /* Allocate memory. */
    rating = malloc(n * m * sizeof(*rating));

    /* Check for success. */
    assert(rating);

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        fscanf(fp, "%lf", &rating[i * m + j]);
      }
    }

    /* Close file. */
    ret = fclose(fp);
    assert(!ret);
  }

  /* Send number of viewers and movies to rest of processes. */
  if (0 == rank) {
    for (int r = 1; r < p; r++) {
      ret = MPI_Send(&n, 1, MPI_SIZE_T, r, 0, MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
      ret = MPI_Send(&m, 1, MPI_SIZE_T, r, 0, MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
    }
  } else {
      ret = MPI_Recv(&n, 1, MPI_SIZE_T, 0, 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
      assert(MPI_SUCCESS == ret);
      ret = MPI_Recv(&m, 1, MPI_SIZE_T, 0, 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
      assert(MPI_SUCCESS == ret);
  }

  /* Compute base number of viewers. */
  size_t const base = 1 + ((n - 1) / p); // ceil(n / p)

  /* Compute local number of viewers. */
  size_t const ln = (rank + 1) * base > n ? n - rank * base : base;

  /* Send viewer data to rest of processes. */
  if (0 == rank) {
    for (int r = 1; r < p; r++) {
      size_t const rn = (r + 1) * base > n ? n - r * base : base;
      ret = MPI_Send(rating + r * base * m, rn * m, MPI_DOUBLE, r, 0,
        MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
    }
  } else {
    /* Allocate memory. */
    rating = malloc(ln * m * sizeof(*rating));

    /* Check for success. */
    assert(rating);

    ret = MPI_Recv(rating, ln * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
    assert(MPI_SUCCESS == ret);
  }

  /* Allocate more memory. */
  double * const urating = malloc((m - 1) * sizeof(*urating));

  /* Check for success. */
  assert(urating);

  /* Get user input and send it to rest of processes. */
  if (0 == rank) {
    for (size_t j = 0; j < m - 1; j++) {
      printf("Enter your rating for movie %zu: ", j + 1);
      fflush(stdout);
      scanf("%lf", &urating[j]);
    }

    for (int r = 1; r < p; r++) {
      ret = MPI_Send(urating, m - 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
    }
  } else {
    ret = MPI_Recv(urating, m - 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
    assert(MPI_SUCCESS == ret);
  }

  /*
  todo:
  Adjust the calculation of distances so that each process is only computing
  the distances for which it is responsible. After completing their distance
  calculations, each process should send their distances back to rank 0,
  which will collect all of the results, sort them, and make a prediction.
  In other words, the code from qsort() onward should only be executed by rank 0.
  */
  if(0 != rank)
  {
    /* Allocate more memory... should now be ln */
    struct distance_metric * const distance = calloc(ln, sizeof(*distance));

    /* Check for success. */
    assert(distance);

    /* Compute distances. */

      for (size_t j = 0; j < ln; j++) {
        distance[ln * rank + j].viewer_id = ln * rank + j;
        //if we are in one of first 4 elements...
        for(size_t k = 0; k < m-1; k++)
        {
          distance[j].distance += fabs(urating[k] - rating[j * m + k]);
        }
        //printf("RANK %d, index %d | input val: %lf | rating val: %lf | abs sum: %lf\n",rank,j,urating[j],rating[i + j],distance[i].distance);
      //printf("RANK %d comp for viewer id %zu: %lf\n",rank,distance[j].viewer_id,distance[j].distance);
      //fflush(stdout);
    }

      //send over distance computations
      ret = MPI_Send(distance, ln , MPI_DISTANCE, 0, 0, MPI_COMM_WORLD);
      assert(MPI_SUCCESS == ret);
  }

  if(0 == rank)
  {
    /* Allocate more memory... should now be n * ln */
    struct distance_metric * const distance = calloc(n, sizeof(*distance));

    /* Check for success. */
    assert(distance);

    for (int r = 1; r < p; r++) {
      ret = MPI_Recv(distance + r * ln, ln , MPI_DISTANCE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      assert(MPI_SUCCESS == ret);
    }

    for (int r = 0; r < n; r++) {
      printf("rating %lf viewer id %zu\n", distance[r].distance, distance[r].viewer_id);
    }


  /* Sort distances. */
  qsort(distance, n, sizeof(*distance), cmp);



  /* Get user input. */
  printf("Enter the number of similar viewers to report: ");
  fflush(stdout);
  scanf("%zu", &k);

  /* Output k viewers who are least different from the user. */
  printf("Viewer ID   Movie five   Distance\n");
  fflush(stdout);
  printf("---------------------------------\n");
  fflush(stdout);

  for (size_t i = 0; i < k; i++) {
    printf("%9zu   %10.1lf   %8.1lf\n", distance[i].viewer_id + 1,
      rating[distance[i].viewer_id * m + 4], distance[i].distance);
    fflush(stdout);
  }

  printf("---------------------------------\n");
  fflush(stdout);

  /* Compute the average to make the prediction. */
  double sum = 0.0;
  for (size_t i = 0; i < k; i++) {
    sum += rating[distance[i].viewer_id * m + 4];
  }

  /* Output prediction. */
  printf("The predicted rating for movie five is %.1lf.\n", sum / k);
  fflush(stdout);
  /* Deallocate memory. */
}

  free(rating);
  free(urating);
  //free(distance);
  ret = MPI_Finalize();
  assert(MPI_SUCCESS == ret);
  return EXIT_SUCCESS;
}
