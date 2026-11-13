// Full-output harness for dtype-specific raised Box/Gaussian C operations.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PVA_FILTER_CTYPE
#error "define PVA_FILTER_CTYPE"
#endif
#ifndef PVA_FILTER_FUNCTION
#error "define PVA_FILTER_FUNCTION"
#endif
#ifndef PVA_FILTER_DTYPE
#error "define PVA_FILTER_DTYPE"
#endif
#ifndef PVA_FILTER_KIND
#error "define PVA_FILTER_KIND: 1=box, 2=gaussian"
#endif
#ifndef PVA_FILTER_TOLERANCE
#define PVA_FILTER_TOLERANCE 0
#endif

typedef PVA_FILTER_CTYPE pixel_t;
enum { H = 192, W = 256, PIXELS = H * W };
#define STRINGIFY_INNER(value) #value
#define STRINGIFY(value) STRINGIFY_INNER(value)

#ifdef PVA_FILTER_FIXED_SHAPE
void PVA_FILTER_FUNCTION(const pixel_t [H][W], pixel_t [H][W]);
#else
void PVA_FILTER_FUNCTION(int, int, const pixel_t *, pixel_t *);
#endif

static uint64_t checksum(const pixel_t *data) {
  const uint8_t *bytes = (const uint8_t *)data;
  uint64_t value = 1469598103934665603ULL;
  for (size_t i = 0; i < sizeof(pixel_t) * PIXELS; ++i) {
    value ^= bytes[i];
    value *= 1099511628211ULL;
  }
  return value;
}

int main(void) {
  enum { GUARD_BYTES = 64 };
  size_t bytes = sizeof(pixel_t) * PIXELS;
  unsigned char *input_storage = malloc(bytes + 2 * GUARD_BYTES);
  unsigned char *output_storage = malloc(bytes + 2 * GUARD_BYTES);
  pixel_t *saved_input = malloc(bytes);
  pixel_t *reference = malloc(sizeof(pixel_t) * PIXELS);
  if (!input_storage || !output_storage || !saved_input || !reference) return 2;
  memset(input_storage, 0xA5, bytes + 2 * GUARD_BYTES);
  memset(output_storage, 0xA5, bytes + 2 * GUARD_BYTES);
  pixel_t *input = (pixel_t *)(input_storage + GUARD_BYTES);
  pixel_t *output = (pixel_t *)(output_storage + GUARD_BYTES);
  const char *pattern = getenv("PVA_FILTER_INPUT_PATTERN");
  if (!pattern) pattern = "original";
  if (strcmp(pattern, "original") && strcmp(pattern, "random") &&
      strcmp(pattern, "impulse") && strcmp(pattern, "constant") &&
      strcmp(pattern, "quantization")) return 2;
  uint32_t state = 12345;
  for (size_t i = 0; i < PIXELS; ++i) {
    state = state * 1664525u + 1013904223u;
    if (!strcmp(pattern, "random")) input[i] = (pixel_t)(state >> 24);
    else if (!strcmp(pattern, "constant")) input[i] = 73;
    else if (!strcmp(pattern, "impulse")) input[i] = i == (H/2)*W+W/2 ? 255 : 0;
    else if (!strcmp(pattern, "quantization")) {
      size_t tap = ((i/W)%3)*3 + (i%W)%3;
      input[i] = tap == 0 || tap == 2 || tap == 4 || tap == 6 ? 255 : 0;
    } else input[i] = (pixel_t)(i * 40503u + i / 17u + 0x9e37u);
  }
  memcpy(saved_input, input, bytes);
  memcpy(output, input, sizeof(pixel_t) * PIXELS);
  memcpy(reference, input, sizeof(pixel_t) * PIXELS);
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x) {
      int sum = 0;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
#if PVA_FILTER_KIND == 1
          sum += input[(y + ky) * W + x + kx];
#elif PVA_FILTER_KIND == 2
          const int weight[3] = {1, 2, 1};
          sum += weight[ky + 1] * weight[kx + 1] *
                 input[(y + ky) * W + x + kx];
#else
#error "unsupported PVA_FILTER_KIND"
#endif
        }
#if PVA_FILTER_KIND == 1
      reference[y * W + x] = (pixel_t)((sum + 4) / 9);
#else
      reference[y * W + x] = (pixel_t)((sum + 8) >> 4);
#endif
    }
#ifdef PVA_FILTER_FIXED_SHAPE
  PVA_FILTER_FUNCTION((const pixel_t (*)[W])input, (pixel_t (*)[W])output);
#else
  PVA_FILTER_FUNCTION(H, W, input, output);
#endif
  size_t mismatches = 0;
  unsigned long long max_error = 0;
  int border_unchanged = 1, guards_ok = 1;
  for (size_t i = 0; i < PIXELS; ++i) {
    long long difference = (long long)output[i] - (long long)reference[i];
    unsigned long long error =
        (unsigned long long)(difference < 0 ? -difference : difference);
    mismatches += error != 0;
    if (error > max_error) max_error = error;
    if ((i/W == 0 || i/W == H-1 || i%W == 0 || i%W == W-1) && error)
      border_unchanged = 0;
  }
  int input_unchanged = memcmp(input, saved_input, bytes) == 0;
  for (size_t i = 0; i < GUARD_BYTES; ++i)
    if (input_storage[i] != 0xA5 || output_storage[i] != 0xA5 ||
        input_storage[GUARD_BYTES+bytes+i] != 0xA5 ||
        output_storage[GUARD_BYTES+bytes+i] != 0xA5) guards_ok = 0;
  const char *status = max_error == 0 ? "EXACT_PASS" :
      max_error <= PVA_FILTER_TOLERANCE ? "WITHIN_TOLERANCE" : "FAIL";
  printf("FILTER_RAISED,%s,%s,%s,mismatches=%zu,max_error=%llu,tolerance=%d,checksum=%llu,"
         "input_unchanged=%d,border_unchanged=%d,guards_ok=%d,input=%s\n",
         PVA_FILTER_KIND == 1 ? "box" : "gaussian",
         STRINGIFY(PVA_FILTER_DTYPE), status, mismatches, max_error,
         PVA_FILTER_TOLERANCE, (unsigned long long)checksum(output),
         input_unchanged, border_unchanged, guards_ok, pattern);
  free(input_storage); free(output_storage); free(reference); free(saved_input);
  return max_error <= PVA_FILTER_TOLERANCE && input_unchanged && border_unchanged && guards_ok ? 0 : 1;
}
