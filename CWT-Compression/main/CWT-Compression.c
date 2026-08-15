#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#define UART_NUM UART_NUM_0
#define BUF_SIZE (2048)
#define TARGET_FS 8000.0f
#define MAX_DWT_LEVEL 5
#define HIST_BUCKETS 4096

static const float h_lp[8] = {0.23037781f, 0.71484657f, 0.63088076f, -0.02798376f,
                               -0.18703481f, 0.03084138f, 0.03288301f, -0.01059740f};
static float h_hp[8];

static void init_filters(void) {
    for (int n = 0; n < 8; n++) {
        h_hp[n] = ((n % 2) ? -1.0f : 1.0f) * h_lp[7 - n];
    }
}

// ---------------------------------------------------------------
// UART helpers
// ---------------------------------------------------------------

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
// DWT building blocks
// ---------------------------------------------------------------

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
        if (!approx || !detail) { free(cur); free(approx); free(detail); return false; }
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
    if (curLen != N) { free(cur); return false; }
    memcpy(out, cur, N * sizeof(float));
    free(cur);
    return true;
}

// ---------------------------------------------------------------
// Incremental histogram (replaces storing every quantized symbol)
// ---------------------------------------------------------------
// Fixed-size static tables (~32 KB total, allocated once in BSS, not
// on the heap, so this cost is constant regardless of file length —
// unlike the old symDWT[] array which scaled with numSamples).

typedef struct {
    int32_t keys[HIST_BUCKETS];
    int     counts[HIST_BUCKETS];
    int     nkeys;
    long    total;
} Histogram;

static void hist_reset(Histogram *h) {
    h->nkeys = 0;
    h->total = 0;
}

static void hist_add(Histogram *h, const int32_t *symbols, int n) {
    for (int i = 0; i < n; i++) {
        int32_t s = symbols[i];
        int found = -1;
        for (int k = 0; k < h->nkeys; k++) {
            if (h->keys[k] == s) { found = k; break; }
        }
        if (found >= 0) {
            h->counts[found]++;
        } else if (h->nkeys < HIST_BUCKETS) {
            h->keys[h->nkeys] = s;
            h->counts[h->nkeys] = 1;
            h->nkeys++;
        }
        h->total++;
    }
}

static float hist_estimate_bits(const Histogram *h) {
    float bits = 0.0f;
    for (int k = 0; k < h->nkeys; k++) {
        float p = (float)h->counts[k] / (float)h->total;
        bits -= h->counts[k] * log2f(p);
    }
    bits += h->nkeys * 24.0f; // rough dictionary overhead
    return bits;
}

// ---------------------------------------------------------------
// Stream-Optimized Pipeline
// ---------------------------------------------------------------
static Histogram g_hist;

static void run_pipeline(void) {
    float step; uint32_t dwtLevel; float frameDuration_ms; uint32_t numSamples;

    if (!read_f32(&step))              { return; }
    if (!read_u32(&dwtLevel))          { return; }
    if (!read_f32(&frameDuration_ms))  { return; }
    if (!read_u32(&numSamples))        { return; }

    if (dwtLevel < 1) dwtLevel = 1;
    if (dwtLevel > MAX_DWT_LEVEL) dwtLevel = MAX_DWT_LEVEL;

    int frameLen = (int)roundf(frameDuration_ms / 1000.0f * TARGET_FS);
    if (frameLen % 2) frameLen++;
    int minDivisor = 1 << dwtLevel;
    frameLen = (frameLen / minDivisor) * minDivisor;
    if (frameLen < minDivisor) frameLen = minDivisor;

    int hopLen = frameLen / 2;
    int numFrames = (numSamples > (uint32_t)frameLen) ? ((numSamples - frameLen) / hopLen) + 1 : 1;
    int padLen = (numFrames - 1) * hopLen + frameLen;

    // Send Handshake to MATLAB
    char ready_msg[64];
    snprintf(ready_msg, sizeof(ready_msg), "READY %d %d %d\n", frameLen, hopLen, numFrames);
    uart_write_bytes(UART_NUM, ready_msg, strlen(ready_msg));

    // Allocate all buffers ONCE before the loop. Constant ~19KB memory footprint.
    float   *raw_frame    = calloc(frameLen, sizeof(float));
    float   *frame        = calloc(frameLen, sizeof(float));
    float   *out_tail     = calloc(hopLen, sizeof(float));
    float   *out_chunk    = calloc(hopLen, sizeof(float));
    float   *hamming      = malloc(frameLen * sizeof(float));
    int32_t *frameSymbols = malloc(frameLen * sizeof(int32_t));
    float   *frDWT        = malloc(frameLen * sizeof(float));

    if (!raw_frame || !frame || !out_tail || !out_chunk || !hamming || !frameSymbols || !frDWT) {
        uart_write_bytes(UART_NUM, "ERR_OOM_BUFFERS\n", 16);
        goto cleanup;
    }

    for (int i = 0; i < frameLen; i++) {
        hamming[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (frameLen - 1));
    }

    hist_reset(&g_hist);

    double sumSqErr = 0, sumSqSig = 0;
    int64_t t0, tPre = 0, tTrans = 0, tComp = 0, tDeco = 0, tReco = 0;

    for (int f = 0; f < numFrames; f++) {
        t0 = esp_timer_get_time();
        
        // --- Stream In ---
        if (f == 0) {
            if (!read_exact((uint8_t*)raw_frame, frameLen * sizeof(float), pdMS_TO_TICKS(15000))) goto cleanup;
        } else {
            memmove(raw_frame, raw_frame + hopLen, hopLen * sizeof(float));
            if (!read_exact((uint8_t*)(raw_frame + hopLen), hopLen * sizeof(float), pdMS_TO_TICKS(5000))) goto cleanup;
        }

        // Apply hamming
        for (int i = 0; i < frameLen; i++) frame[i] = raw_frame[i] * hamming[i];
        tPre += esp_timer_get_time() - t0;

        // --- Forward DWT ---
        t0 = esp_timer_get_time();
        DwtResult dwtR;
        if (!dwt_multilevel(frame, frameLen, dwtLevel, &dwtR)) {
            uart_write_bytes(UART_NUM, "ERR_OOM_DWT\n", 12);
            goto cleanup;
        }
        tTrans += esp_timer_get_time() - t0;

        // --- Quantize & Hist ---
        t0 = esp_timer_get_time();
        int wpos = 0;
        for (int i = 0; i < dwtR.approx_len; i++) frameSymbols[wpos++] = (int32_t)roundf(dwtR.approx_final[i] / step);
        for (int lvl = dwtR.levels - 1; lvl >= 0; lvl--)
            for (int i = 0; i < dwtR.detail_len[lvl]; i++)
                frameSymbols[wpos++] = (int32_t)roundf(dwtR.detail[lvl][i] / step);
        hist_add(&g_hist, frameSymbols, wpos);
        tComp += esp_timer_get_time() - t0;

        // --- Dequantize ---
        t0 = esp_timer_get_time();
        wpos = 0;
        for (int i = 0; i < dwtR.approx_len; i++) dwtR.approx_final[i] = frameSymbols[wpos++] * step;
        for (int lvl = dwtR.levels - 1; lvl >= 0; lvl--)
            for (int i = 0; i < dwtR.detail_len[lvl]; i++)
                dwtR.detail[lvl][i] = frameSymbols[wpos++] * step;
        tDeco += esp_timer_get_time() - t0;

        // --- Inverse DWT ---
        t0 = esp_timer_get_time();
        dwt_inverse_multilevel(&dwtR, frDWT, frameLen);
        tReco += esp_timer_get_time() - t0;

        // --- Overlap Add & Stream Out ---
        for (int i = 0; i < hopLen; i++) {
            out_chunk[i] = frDWT[i] + out_tail[i]; // Combine current frame half with previous tail
            out_tail[i]  = frDWT[i + hopLen];      // Cache second half for next frame's tail
            
            // Rough metrics calc
            float orig = raw_frame[i];
            sumSqSig += (double)orig * orig;
            sumSqErr += (double)(orig - out_chunk[i]) * (orig - out_chunk[i]);
        }
        
        uart_write_bytes(UART_NUM, (const char*)out_chunk, hopLen * sizeof(float));
        dwt_free(&dwtR);
    }

    // Write the final overlap-added tail
    uart_write_bytes(UART_NUM, (const char*)out_tail, hopLen * sizeof(float));

    // --- Metrics & Finalization ---
    float snr = (sumSqErr > 0) ? 10.0f * log10f(sumSqSig / sumSqErr) : INFINITY;
    float rmse = sqrtf(sumSqErr / padLen);

    float est_bits = hist_estimate_bits(&g_hist);
    float origBits = (float)(padLen) * 16.0f;
    float cr = (est_bits > 0.1f) ? (origBits / est_bits) : 1.0f;

    float preMs = tPre / 1000.0f, transMs = tTrans / 1000.0f, compMs = tComp / 1000.0f,
          decoMs = tDeco / 1000.0f, recoMs = tReco / 1000.0f;
    float metrics[8] = { cr, snr, rmse, preMs, transMs, compMs, decoMs, recoMs };

    uart_write_bytes(UART_NUM, "DONE\n", 5);
    uart_write_bytes(UART_NUM, (const char*)metrics, sizeof(metrics));

cleanup:
    free(raw_frame); free(frame); free(out_tail); free(out_chunk);
    free(hamming); free(frameSymbols); free(frDWT);
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
    }
}

void app_main(void) {
    esp_log_level_set("*", ESP_LOG_NONE);
    xTaskCreatePinnedToCore(dsp_processing_task, "DSP_Task", 16384, NULL, 10, NULL, 1);
}