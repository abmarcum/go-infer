#ifndef INFERENCE_CORE_H
#define INFERENCE_CORE_H

#include <stddef.h>

typedef struct {
    char* model_path;
    int is_loaded;
    int has_metal;
    char* device_name;
    void* metal_device;
} ModelContext;

ModelContext* init_model(const char* path);
char* generate_tokens(ModelContext* ctx, const char* prompt);
void free_model(ModelContext* ctx);
int model_has_metal(const ModelContext* ctx);
const char* model_device_name(const ModelContext* ctx);

#endif
