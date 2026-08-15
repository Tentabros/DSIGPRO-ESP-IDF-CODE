#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_timer.h"
#include "esp_dsp.h"

// Define Hardware & DSP Parameters
#define UART_NUM UART_NUM_0
#define BUF_SIZE (1024)
#define FRAME_LEN 200 // 25 ms frame at 8000 Hz target sampling rate

// Daubechies 4 (db4) Filter Coefficients
const float h_db4[8] = {0.23037781, 0.71484657, 0.63088076, -0.02798376, -0.18703481, 0.03084138, 0.03288301, -0.01059740};

// Packet Structure for UART Transmission
typedef struct __attribute__((packed)) {
    int64_t processing_time_us;
    float quantized_coeffs[FRAME_LEN / 2];
} DataPacket;

void dsp_processing_task(void *arg) {
    // 1. Configure UART for PC Communication
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE
    };
    uart_param_config(UART_NUM, &uart_config);
    uart_set_pin(UART_NUM, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);

    float frame_buffer[FRAME_LEN];
    float dct_output[FRAME_LEN / 2];
    DataPacket tx_packet;

    while (1) {
        // Read incoming raw audio frame from MATLAB
        int len = uart_read_bytes(UART_NUM, (uint8_t*)frame_buffer, sizeof(frame_buffer), portMAX_DELAY);
        
        if (len == sizeof(frame_buffer)) {
            // Start hardware timer interrupt
            int64_t start_time = esp_timer_get_time();
            
            // --- ALGORITHM START ---
            
            // Step 1: 1-Level DWT (Low-Pass Approximation)
            float approx[FRAME_LEN / 2];
            for (int i = 0; i < FRAME_LEN / 2; i++) {
                approx[i] = 0;
                for (int j = 0; j < 8; j++) {
                    int idx = (2 * i + j) % FRAME_LEN; // Circular convolution
                    approx[i] += frame_buffer[idx] * h_db4[j];
                }
            }

            // Step 2: DCT Energy Compaction
            // ESP-DSP DCT is an in-place function: esp_err_t dsps_dct_f32(float *data, int N);
            // Copy 'approx' to 'dct_output' first, then transform it in-place.
            for (int i = 0; i < FRAME_LEN / 2; i++) {
                dct_output[i] = approx[i];
            }
            dsps_dct_f32(dct_output, FRAME_LEN / 2);

            // Step 3: Quantization
            float step_size = 0.01;
            for(int i = 0; i < FRAME_LEN / 2; i++) {
                tx_packet.quantized_coeffs[i] = (int)(dct_output[i] / step_size);
            }

            // --- ALGORITHM END ---
            
            // Record PT and send packet to PC
            tx_packet.processing_time_us = esp_timer_get_time() - start_time;
            uart_write_bytes(UART_NUM, (const char*)&tx_packet, sizeof(DataPacket));
        }
    }
}

void app_main(void) {
    // Pin to Core 1 to avoid OS interrupt conflicts
    xTaskCreatePinnedToCore(dsp_processing_task, "DSP_Task", 8192, NULL, 10, NULL, 1);
}