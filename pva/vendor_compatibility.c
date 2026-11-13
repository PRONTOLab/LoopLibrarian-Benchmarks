/* Compare the unchanged fixture functions with real PVA APIs. This diagnostic
 * does not authorize a compiler replacement or count as a matching result. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fixture_box_filter3x3(int,int,const uint8_t *,uint8_t *);
void fixture_bilateral_filter3x3(int,int,const uint8_t *,uint8_t *,float,float);
void fixture_histogram_equalization(int,int,const uint8_t *,uint8_t *);
void polygeist_pva_boxfilter_3x3_u8(int32_t,int32_t,const uint8_t *,uint8_t *);
void polygeist_pva_bilateral_3x3_u8(int32_t,int32_t,float,float,const uint8_t *,uint8_t *);
void polygeist_pva_histeq_u8(int32_t,int32_t,const uint8_t *,uint8_t *);

int main(void) {
  enum {H=192,W=256,N=H*W};
  uint8_t *in=malloc(N),*saved=malloc(N),*ref=malloc(N),*out=malloc(N+32);
  if(!in||!saved||!ref||!out)return 2;
  const char *names[]={"BoxFilter","BilateralFilter","HistogramEqualization"};
  const char *recipes[]={"original_filter","random","impulse","constant","quantization_extremes"};
  for(int mode=0;mode<5;++mode) {
    uint32_t state=0x12345678u;
    for(int i=0;i<N;++i) {
      state^=state<<13;state^=state>>17;state^=state<<5;
      in[i]=mode==0?(uint8_t)(i*40503u+i/17u+0x9e37u):
        mode==1?(uint8_t)state:mode==2?(i==N/2?255:0):mode==3?73:0;
    }
    if(mode==4) {
      /* Public BoxFilter Q8 coefficient adjustment increments offsets
       * 0,2,4,6 of the 3x3 kernel. Maximize those input pixels to expose
       * error hidden by random tests; this is an input, not a new oracle. */
      for(int k=0;k<8;k+=2)in[(H/2-1+k/3)*W+W/2-1+k%3]=255;
    }
    memcpy(saved,in,N);
    for(int op=0;op<3;++op) {
      memcpy(ref,in,N);memcpy(out+16,in,N);
      memset(out,0xa5,16);memset(out+16+N,0x5a,16);
      if(op==0) {
        fixture_box_filter3x3(H,W,in,ref);
        polygeist_pva_boxfilter_3x3_u8(H,W,in,out+16);
      } else if(op==1) {
        fixture_bilateral_filter3x3(H,W,in,ref,25.0f,1.5f);
        polygeist_pva_bilateral_3x3_u8(H,W,25.0f,1.5f,in,out+16);
      } else {
        fixture_histogram_equalization(H,W,in,ref);
        polygeist_pva_histeq_u8(H,W,in,out+16);
      }
      int mismatches=0,maximum=0,guardbad=0;
      for(int i=0;i<N;++i) {
        int error=abs((int)ref[i]-(int)out[i+16]);
        mismatches+=error!=0;if(error>maximum)maximum=error;
      }
      for(int i=0;i<16;++i)guardbad+=out[i]!=0xa5||out[N+16+i]!=0x5a;
      int inputbad=memcmp(in,saved,N)!=0;
      printf("COMPARE,%s,%s,count=%d,mismatches=%d,max_error=%d,guardbad=%d,inputbad=%d\n",
        names[op],recipes[mode],N,mismatches,maximum,guardbad,inputbad);
      fflush(stdout);
      if(guardbad||inputbad)return 2;
    }
  }
  free(in);free(saved);free(ref);free(out);
  return 0; /* Collection completed; mismatches remain explicit in each row. */
}
