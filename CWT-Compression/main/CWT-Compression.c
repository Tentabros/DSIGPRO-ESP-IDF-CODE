#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_log.h"

#define UART_NUM UART_NUM_0
#define BUF_SIZE (2048)
#define TARGET_FS 8000.0f
#define MAX_DWT_LEVEL 5

// Daubechies 4 (db4) analysis low-pass filter
static const float h_lp[8] = {0.23037781f, 0.71484657f, 0.63088076f, -0.02798376f,
                               -0.18703481f, 0.03084138f, 0.03288301f, -0.01059740f};
// Derived analysis high-pass via QMF relation: g[n] = (-1)^n * h[7-n]
static float h_hp[8];

static void init_filters(void) {
    for (int n = 0; n < 8; n++) {
        h_hp[n] = ((n % 2) ? -1.0f : 1.0f) * h_lp[7 - n];
    }
}

// ---------------------------------------------------------------
// UART helpers
// ---------------------------------------------------------------

// Blocking read of exactly `len` bytes (uart_read_bytes can return short reads)
static bool read_exact(uint8_t *dst, size_t len, TickType_t timeout_ticks) {
    size_t got = 0;
    while (got < len) {
        int n = uart_read_bytes(UART_NUM, dst + got, len - got, timeout_ticks);
        if (n <= 0) return false;
        got += n;
    }
    return true;
}

static bool read_line(char *dst, size_t maxlen) {
    size_t i = 0;
    while (i < maxlen - 1) {
        uint8_t c;
        int n = uart_read_bytes(UART_NUM, &c, 1, pdMS_TO_TICKS(5000));
        if (n <= 0) return false;
        if (c == '\n') { dst[i] = '\0'; return true; }
        if (c != '\r') dst[i++] = (char)c;
    }
    dst[i] = '\0';
    return true;
}

static bool read_f32(float *v) { return read_exact((uint8_t*)v, 4, pdMS_TO_TICKS(5000)); }
static bool read_u32(uint32_t *v) { return read_exact((uint8_t*)v, 4, pdMS_TO_TICKS(5000)); }

// ---------------------------------------------------------------
// DSP building blocks
// ---------------------------------------------------------------

// Naive O(N^2) DCT-II (matches MATLAB's dct() normalization)
static void dct_forward(const float *in, float *out, int N) {
    for (int k = 0; k < N; k++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += in[n] * cosf((float)M_PI / N * (n + 0.5f) * k);
        }
        float scale = (k == 0) ? sqrtf(1.0f / N) : sqrtf(2.0f / N);
        out[k] = scale * sum;
    }
}

static void dct_inverse(const float *in, float *out, int N) {
    for (int n = 0; n < N; n++) {
        float sum = in[0] * sqrtf(1.0f / N);
        for (int k = 1; k < N; k++) {
            sum += in[k] * sqrtf(2.0f / N) * cosf((float)M_PI / N * (n + 0.5f) * k);
        }
        out[n] = sum;
    }
}

// One level of DWT decomposition (periodic/circular extension, matches your original approach)
static void dwt_step(const float *in, int N, float *approx, float *detail) {
    int half = N / 2;
    for (int i = 0; i < half; i++) {
        float a = 0.0f, d = 0.0f;
        for (int j = 0; j < 8; j++) {
            int idx = (2 * i + j) % N;
            a += in[idx] * h_lp[j];
            d += in[idx] * h_hp[j];
        }
        approx[i] = a;
        detail[i] = d;
    }
}

// One level of inverse DWT (orthogonal filter -> reconstruction filters are
// time-reversed analysis filters)
static void idwt_step(const float *approx, const float *detail, int half, float *out) {
    int N = half * 2;
    for (int i = 0; i < N; i++) out[i] = 0.0f;
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < 8; j++) {
            int idx = (2 * i + j) % N;
            out[idx] += approx[i] * h_lp[7 - j] + detail[i] * h_hp[7 - j];
        }
    }
}

// Multi-level DWT: recursively decompose the approximation, matching
// MATLAB's wavedec (concatenated as [cAn, cDn, ..., cD1] conceptually).
// We keep it simpler internally: store each level's approx/detail directly.
typedef struct {
    float *detail[MAX_DWT_LEVEL];
    int    detail_len[MAX_DWT_LEVEL];
    float *approx_final;
    int    approx_len;
    int    levels;
} DwtResult;

static bool dwt_multilevel(const float *frame, int N, int levels, DwtResult *r) {
    r->levels = levels;
    float *cur = (float*)malloc(N * sizeof(float));
    if (!cur) return false;
    memcpy(cur, frame, N * sizeof(float));
    int curLen = N;

    for (int lvl = 0; lvl < levels; lvl++) {
        int half = curLen / 2;
        float *approx = (float*)malloc(half * sizeof(float));
        float *detail = (float*)malloc(half * sizeof(float));
        if (!approx || !detail) { free(cur); return false; }
        dwt_step(cur, curLen, approx, detail);
        r->detail[lvl] = detail;
        r->detail_len[lvl] = half;
        free(cur);
        cur = approx;
        curLen = half;
    }
    r->approx_final = cur;
    r->approx_len = curLen;
    return true;
}

static void dwt_free(DwtResult *r) {
    free(r->approx_final);
    for (int i = 0; i < r->levels; i++) free(r->detail[i]);
}

static bool dwt_inverse_multilevel(DwtResult *r, float *out, int N) {
    float *cur = (float*)malloc(r->approx_len * sizeof(float));
    if (!cur) return false;
    memcpy(cur, r->approx_final, r->approx_len * sizeof(float));
    int curLen = r->approx_len;

    for (int lvl = r->levels - 1; lvl >= 0; lvl--) {
        int outLen = curLen * 2;
        float *rec = (float*)malloc(outLen * sizeof(float));
        if (!rec) { free(cur); return false; }
        idwt_step(cur, r->detail[lvl], curLen, rec);
        free(cur);
        cur = rec;
        curLen = outLen;
    }
    if (curLen != N) { free(cur); return false; } // sanity check
    memcpy(out, cur, N * sizeof(float));
    free(cur);
    return true;
}

// Quantize in place, return count of distinct symbols + a zero-order
// entropy estimate (used as a CR proxy — this is NOT a bit-exact match
// to MATLAB's Huffman bitLength, just a reasonable approximation).
static float estimate_compressed_bits(const int32_t *symbols, int n) {
    // Simple histogram-based Shannon entropy estimate
    // (bounded symbol range assumption; fine for quantized audio coeffs)
    #define HIST_BUCKETS 4096
    static int32_t keys[HIST_BUCKETS];
    static int counts[HIST_BUCKETS];
    int nkeys = 0;

    for (int i = 0; i < n; i++) {
        int32_t s = symbols[i];
        int found = -1;
        for (int k = 0; k < nkeys; k++) {
            if (keys[k] == s) { found = k; break; }
        }
        if (found >= 0) {
            counts[found]++;
        } else if (nkeys < HIST_BUCKETS) {
            keys[nkeys] = s;
            counts[nkeys] = 1;
            nkeys++;
        }
    }
    float bits = 0.0f;
    for (int k = 0; k < nkeys; k++) {
        float p = (float)counts[k] / n;
        bits -= counts[k] * log2f(p);
    }
    // rough dictionary overhead: ~ (16 bits code length + symbol) per unique symbol
    bits += nkeys * 24.0f;
    return bits;
}

// ---------------------------------------------------------------
// Pipeline for one command (shared by PROCESS_SINGLE / PROCESS_BATCH)
// ---------------------------------------------------------------

static void run_pipeline(void) {
    float step; uint32_t dwtLevel; float frameDuration_ms; uint32_t numSamples;

    // Declare ALL heap pointers up front, NULL-initialized, so that any
    // `goto cleanup_fail` below is well-defined no matter which allocation
    // step it happens after. This is what fixes the -Werror=maybe-uninitialized
    // build failure — free(NULL) is always a safe no-op.
    float   *audio  = NULL;
    float   *hamming = NULL;
    float   *recDCT = NULL;
    float   *recDWT = NULL;
    float   *recHYB = NULL;
    float   *frame  = NULL;
    int32_t *symDCT = NULL;
    int32_t *symDWT = NULL;
    int32_t *symHYB = NULL;

    if (!read_f32(&step))              { return; }
    if (!read_u32(&dwtLevel))          { return; }
    if (!read_f32(&frameDuration_ms))  { return; }
    if (!read_u32(&numSamples))        { return; }

    if (dwtLevel < 1) dwtLevel = 1;
    if (dwtLevel > MAX_DWT_LEVEL) dwtLevel = MAX_DWT_LEVEL;

    audio = (float*)malloc(numSamples * sizeof(float));
    if (!audio) { uart_write_bytes(UART_NUM, "DONE\n", 5); return; }

    if (!read_exact((uint8_t*)audio, numSamples * sizeof(float), pdMS_TO_TICKS(30000))) {
        free(audio);
        return;
    }

    int frameLen = (int)roundf(frameDuration_ms / 1000.0f * TARGET_FS);
    if (frameLen % 2) frameLen++;
    int minDivisor = 1 << dwtLevel;
    frameLen = (frameLen / minDivisor) * minDivisor;
    if (frameLen < minDivisor) frameLen = minDivisor;

    int hopLen = frameLen / 2;
    int numFrames = (numSamples > (uint32_t)frameLen) ? ((numSamples - frameLen) / hopLen) + 1 : 1;
    int padLen = (numFrames - 1) * hopLen + frameLen;

    hamming = (float*)malloc(frameLen * sizeof(float));
    recDCT  = (float*)calloc(padLen, sizeof(float));
    recDWT  = (float*)calloc(padLen, sizeof(float));
    recHYB  = (float*)calloc(padLen, sizeof(float));
    frame   = (float*)malloc(frameLen * sizeof(float));
    if (!hamming || !recDCT || !recDWT || !recHYB || !frame) goto cleanup_fail;

    for (int i = 0; i < frameLen; i++) {
        hamming[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (frameLen - 1));
    }

    double sumSqErrDCT = 0, sumSqErrDWT = 0, sumSqErrHYB = 0, sumSqSig = 0;
    symDCT = (int32_t*)malloc(frameLen * numFrames * sizeof(int32_t));
    symDWT = (int32_t*)malloc(frameLen * numFrames * sizeof(int32_t));
    symHYB = (int32_t*)malloc(frameLen * numFrames * sizeof(int32_t));
    if (!symDCT || !symDWT || !symHYB) goto cleanup_fail;
    int symIdx = 0;

    int64_t t0, tPre = 0, tTrans = 0, tComp = 0, tDeco = 0, tReco = 0;

    for (int f = 0; f < numFrames; f++) {
        t0 = esp_timer_get_time();
        int start = f * hopLen;
        for (int i = 0; i < frameLen; i++) {
            int idx = start + i;
            float s = (idx < (int)numSamples) ? audio[idx] : 0.0f;
            frame[i] = s * hamming[i];
        }
        tPre += esp_timer_get_time() - t0;

        // --- Transform ---
        t0 = esp_timer_get_time();
        float *dctC = (float*)malloc(frameLen * sizeof(float));
        DwtResult dwtR;
        dwt_multilevel(frame, frameLen, dwtLevel, &dwtR);
        dct_forward(frame, dctC, frameLen);
        tTrans += esp_timer_get_time() - t0;

        // Hybrid: DCT applied to the DWT approximation coefficients
        t0 = esp_timer_get_time();
        float *hybC = (float*)malloc(dwtR.approx_len * sizeof(float));
        dct_forward(dwtR.approx_final, hybC, dwtR.approx_len);
        tTrans += esp_timer_get_time() - t0;

        // --- Compress (quantize) ---
        t0 = esp_timer_get_time();
        for (int i = 0; i < frameLen; i++) symDCT[symIdx + i] = (int32_t)roundf(dctC[i] / step);
        for (int i = 0; i < dwtR.approx_len; i++) symHYB[symIdx + i] = (int32_t)roundf(hybC[i] / step);
        int wpos = 0;
        int32_t *dwtSymFrame = (int32_t*)malloc(frameLen * sizeof(int32_t));
        for (int i = 0; i < dwtR.approx_len; i++) dwtSymFrame[wpos++] = (int32_t)roundf(dwtR.approx_final[i] / step);
        for (int lvl = dwtR.levels - 1; lvl >= 0; lvl--)
            for (int i = 0; i < dwtR.detail_len[lvl]; i++)
                dwtSymFrame[wpos++] = (int32_t)roundf(dwtR.detail[lvl][i] / step);
        memcpy(&symDWT[symIdx], dwtSymFrame, wpos * sizeof(int32_t));
        tComp += esp_timer_get_time() - t0;

        // --- Dequantize ---
        t0 = esp_timer_get_time();
        float *dctD = (float*)malloc(frameLen * sizeof(float));
        for (int i = 0; i < frameLen; i++) dctD[i] = symDCT[symIdx + i] * step;
        wpos = 0;
        for (int i = 0; i < dwtR.approx_len; i++) dwtR.approx_final[i] = dwtSymFrame[wpos++] * step;
        for (int lvl = dwtR.levels - 1; lvl >= 0; lvl--)
            for (int i = 0; i < dwtR.detail_len[lvl]; i++)
                dwtR.detail[lvl][i] = dwtSymFrame[wpos++] * step;
        float *hybD = (float*)malloc(dwtR.approx_len * sizeof(float));
        for (int i = 0; i < dwtR.approx_len; i++) hybD[i] = symHYB[symIdx + i] * step;
        tDeco += esp_timer_get_time() - t0;

        // --- Reconstruct ---
        t0 = esp_timer_get_time();
        float *frDCT = (float*)malloc(frameLen * sizeof(float));
        dct_inverse(dctD, frDCT, frameLen);

        float *frDWT = (float*)malloc(frameLen * sizeof(float));
        dwt_inverse_multilevel(&dwtR, frDWT, frameLen);

        float *hybApproxRec = (float*)malloc(dwtR.approx_len * sizeof(float));
        dct_inverse(hybD, hybApproxRec, dwtR.approx_len);
        DwtResult hybFull = dwtR;
        hybFull.approx_final = hybApproxRec;
        float *frHYB = (float*)malloc(frameLen * sizeof(float));
        dwt_inverse_multilevel(&hybFull, frHYB, frameLen);
        tReco += esp_timer_get_time() - t0;

        for (int i = 0; i < frameLen; i++) {
            recDCT[start + i] += frDCT[i];
            recDWT[start + i] += frDWT[i];
            recHYB[start + i] += frHYB[i];
            float orig = frame[i];
            sumSqSig += (double)orig * orig;
            sumSqErrDCT += (double)(orig - frDCT[i]) * (orig - frDCT[i]);
            sumSqErrDWT += (double)(orig - frDWT[i]) * (orig - frDWT[i]);
            sumSqErrHYB += (double)(orig - frHYB[i]) * (orig - frHYB[i]);
        }

        symIdx += frameLen;
        free(dctC); free(hybC); free(dwtSymFrame); free(dctD);
        free(hybD); free(frDCT); free(frDWT); free(hybApproxRec); free(frHYB);
        dwt_free(&dwtR);
    }

    // --- Metrics ---
    float snrDCT = (sumSqErrDCT > 0) ? 10.0f * log10f(sumSqSig / sumSqErrDCT) : INFINITY;
    float snrDWT = (sumSqErrDWT > 0) ? 10.0f * log10f(sumSqSig / sumSqErrDWT) : INFINITY;
    float snrHYB = (sumSqErrHYB > 0) ? 10.0f * log10f(sumSqSig / sumSqErrHYB) : INFINITY;
    float rmseDCT = sqrtf(sumSqErrDCT / (frameLen * numFrames));
    float rmseDWT = sqrtf(sumSqErrDWT / (frameLen * numFrames));
    float rmseHYB = sqrtf(sumSqErrHYB / (frameLen * numFrames));

    int bitDepth = 16; // assume 16-bit source; adjust if you send bitDepth from MATLAB
    float origBits = (float)(frameLen * numFrames) * bitDepth;
    float crDCT = origBits / estimate_compressed_bits(symDCT, frameLen * numFrames);
    float crDWT = origBits / estimate_compressed_bits(symDWT, frameLen * numFrames);
    float crHYB = origBits / estimate_compressed_bits(symHYB, frameLen * numFrames);

    float preMs = tPre / 1000.0f, transMs = tTrans / 1000.0f, compMs = tComp / 1000.0f,
          decoMs = tDeco / 1000.0f, recoMs = tReco / 1000.0f;

    float metrics[24] = {
        crDCT, crDWT, crHYB,
        snrDCT, snrDWT, snrHYB,
        rmseDCT, rmseDWT, rmseHYB,
        preMs, preMs, preMs,
        transMs, transMs, transMs,
        compMs, compMs, compMs,
        decoMs, decoMs, decoMs,
        recoMs, recoMs, recoMs
    };

    // --- Send response ---
    uart_write_bytes(UART_NUM, "DONE\n", 5);
    uart_write_bytes(UART_NUM, (const char*)metrics, sizeof(metrics));
    uart_write_bytes(UART_NUM, (const char*)recDCT, numSamples * sizeof(float));
    uart_write_bytes(UART_NUM, (const char*)recDWT, numSamples * sizeof(float));
    uart_write_bytes(UART_NUM, (const char*)recHYB, numSamples * sizeof(float));

cleanup_fail:
    free(audio); free(hamming); free(recDCT); free(recDWT); free(recHYB); free(frame);
    free(symDCT); free(symDWT); free(symHYB);
}

// ---------------------------------------------------------------
// Main command loop
// ---------------------------------------------------------------

void dsp_processing_task(void *arg) {
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE
    };
    uart_param_config(UART_NUM, &uart_config);
    uart_set_pin(UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM, BUF_SIZE * 4, BUF_SIZE * 4, 0, NULL, 0);

    init_filters();

    char line[64];
    while (1) {
        if (!read_line(line, sizeof(line))) continue;

        if (strcmp(line, "PING") == 0) {
            uart_write_bytes(UART_NUM, "ACK_ESP32\n", 10);
        } else if (strcmp(line, "PROCESS_SINGLE") == 0 || strcmp(line, "PROCESS_BATCH") == 0) {
            run_pipeline();
        }
        // unrecognized lines are silently ignored, matching MATLAB's readline-until-match loop
    }
}

void app_main(void) {
    esp_log_level_set("*", ESP_LOG_NONE); // keep UART0 clean of console output
    xTaskCreatePinnedToCore(dsp_processing_task, "DSP_Task", 16384, NULL, 10, NULL, 1);
}