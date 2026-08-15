idf_component_register(SRCS "CWT-Compression.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp-dsp driver esp_driver_uart esp_timer)