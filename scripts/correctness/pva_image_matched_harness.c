/* Execute the compiler-generated function and the unchanged source reference.
 * Numeric differences are reported; execution and memory checks decide success. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../pva/inputs/image_processing.c"
typedef PVA_IMAGE_CTYPE pixel_t;
enum {H=192,W=256,N=H*W,G=64};
#if PVA_IMAGE_DYNAMIC_SHAPE
#if PVA_IMAGE_BILATERAL
void PVA_IMAGE_FUNCTION(int,int,const pixel_t *,pixel_t *,float,float);
#else
void PVA_IMAGE_FUNCTION(int,int,const pixel_t *,pixel_t *);
#endif
#else
void PVA_IMAGE_FUNCTION(const pixel_t [H][W],pixel_t [H][W]);
#endif
int main(void) {
  size_t bytes=N*sizeof(pixel_t);
  unsigned char *a=malloc(bytes+2*G),*b=malloc(bytes+2*G);
  pixel_t *saved=malloc(bytes),*reference=malloc(bytes);
  if(!a||!b||!saved||!reference)return 2;
  memset(a,0xa5,bytes+2*G);memset(b,0xa5,bytes+2*G);
  pixel_t *input=(pixel_t*)(a+G),*output=(pixel_t*)(b+G);
  const char *pattern=getenv("PVA_FILTER_INPUT_PATTERN");if(!pattern)pattern="original";
  if(strcmp(pattern,"original")&&strcmp(pattern,"random")&&strcmp(pattern,"impulse")&&
     strcmp(pattern,"constant")&&strcmp(pattern,"quantization"))return 2;
  uint32_t state=0x12345678;
  for(int i=0;i<N;++i) {
    state^=state<<13;state^=state>>17;state^=state<<5;
    input[i]=!strcmp(pattern,"random")?(pixel_t)state:
      !strcmp(pattern,"constant")?73:!strcmp(pattern,"impulse")?(i==N/2?255:0):
      !strcmp(pattern,"quantization")?(pixel_t)((i%3)==0?~0u:0):
      (pixel_t)(i*73u+i/17u+19);
  }
  memcpy(saved,input,bytes);memcpy(output,input,bytes);memcpy(reference,input,bytes);
#if PVA_IMAGE_BILATERAL
  PVA_IMAGE_REFERENCE(H,W,input,reference,25.0f,1.5f);
#else
  PVA_IMAGE_REFERENCE(H,W,input,reference);
#endif
#if PVA_IMAGE_DYNAMIC_SHAPE
#if PVA_IMAGE_BILATERAL
  PVA_IMAGE_FUNCTION(H,W,input,output,25.0f,1.5f);
#else
  PVA_IMAGE_FUNCTION(H,W,input,output);
#endif
#else
  PVA_IMAGE_FUNCTION((const pixel_t (*)[W])input,(pixel_t (*)[W])output);
#endif
  unsigned long long maximum=0;size_t mismatches=0;
  int border=1,guards=1,unchanged=memcmp(input,saved,bytes)==0;
  for(int i=0;i<N;++i) {
    long long d=(long long)output[i]-reference[i];unsigned long long e=d<0?-d:d;
    if(e>maximum)maximum=e;mismatches+=e!=0;
#if PVA_IMAGE_PRESERVE_BORDER
    if((i/W==0||i/W==H-1||i%W==0||i%W==W-1)&&output[i]!=saved[i])border=0;
#endif
  }
  for(int i=0;i<G;++i)if(a[i]!=0xa5||b[i]!=0xa5||a[G+bytes+i]!=0xa5||b[G+bytes+i]!=0xa5)guards=0;
  printf("IMAGE_RAISED,NUMERICS_RECORDED,mismatches=%zu,max_error=%llu,input_unchanged=%d,border_unchanged=%d,guards_ok=%d,input=%s\n",
         mismatches,maximum,unchanged,border,guards,pattern);
  free(a);free(b);free(saved);free(reference);return unchanged&&border&&guards?0:1;
}
