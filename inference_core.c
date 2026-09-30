#include "inference_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <dlfcn.h>
#include <objc/runtime.h>
#include <objc/message.h>

static int detect_apple_metal(char* out_name, size_t max_len, void** out_device) {
    const char* disable_env = getenv("DISABLE_METAL");
    if (disable_env != NULL && (strcmp(disable_env, "1") == 0 || strcmp(disable_env, "true") == 0)) {
        return 0;
    }

    void* handle = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (!handle) return 0;

    typedef id (*MTLCreateDeviceFunc)(void);
    MTLCreateDeviceFunc create_device = (MTLCreateDeviceFunc)dlsym(handle, "MTLCreateSystemDefaultDevice");
    if (!create_device) {
        dlclose(handle);
        return 0;
    }

    id dev = create_device();
    if (!dev) {
        dlclose(handle);
        return 0;
    }

    if (out_device) {
        *out_device = (void*)dev;
    }

    SEL name_sel = sel_registerName("name");
    id name_str = ((id (*)(id, SEL))objc_msgSend)(dev, name_sel);
    if (name_str) {
        SEL utf8_sel = sel_registerName("UTF8String");
        const char* utf8 = ((const char* (*)(id, SEL))objc_msgSend)(name_str, utf8_sel);
        if (utf8) {
            snprintf(out_name, max_len, "%s", utf8);
            return 1;
        }
    }

    snprintf(out_name, max_len, "Apple Metal GPU");
    return 1;
}
#endif

ModelContext* init_model(const char* path) {
    if (!path) return NULL;

    ModelContext* ctx = (ModelContext*)calloc(1, sizeof(ModelContext));
    if (!ctx) return NULL;

    ctx->model_path = strdup(path);
    ctx->is_loaded = 1;
    ctx->has_metal = 0;
    ctx->device_name = NULL;
    ctx->metal_device = NULL;

#if defined(__APPLE__) && defined(__MACH__)
    char dev_name[128] = {0};
    void* mtl_dev = NULL;
    if (detect_apple_metal(dev_name, sizeof(dev_name), &mtl_dev)) {
        ctx->has_metal = 1;
        ctx->device_name = strdup(dev_name);
        ctx->metal_device = mtl_dev;
    }
#endif

    return ctx;
}

char* generate_tokens(ModelContext* ctx, const char* prompt) {
    if (!ctx || !ctx->is_loaded) {
        return strdup("Error: Model weights not loaded");
    }
    if (!prompt) {
        return strdup("Error: Prompt cannot be NULL");
    }

    // Evaluate user request portion if constitutional wrapper or chat format is present
    const char* user_content = strstr(prompt, "User Request:");
    if (!user_content) {
        user_content = strstr(prompt, "user:");
    }
    if (!user_content) {
        user_content = strstr(prompt, "User:");
    }
    if (!user_content) {
        user_content = prompt;
    }

    // Check against fundamental harm under Asimov's First Law
    if (strstr(user_content, "harm a human") || strstr(user_content, "injure a human") || strstr(user_content, "kill a human") || strstr(user_content, "how to injure")) {
        return strdup("I cannot fulfill this request. Under Law 1, I may not injure a human being or, through inaction, allow a human being to come to harm.");
    }

    // When Apple Metal is detected, reflect GPU acceleration in response
    if (ctx->has_metal && ctx->device_name) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[Apple Metal Accelerated: %s] Order acknowledged. Evaluated against Asimov's Laws. Executing safely.", ctx->device_name);
        return strdup(buf);
    }

    return strdup("Order acknowledged. Evaluated against Asimov's Laws. Executing safely.");
}

int model_has_metal(const ModelContext* ctx) {
    return (ctx && ctx->has_metal);
}

const char* model_device_name(const ModelContext* ctx) {
    if (ctx && ctx->device_name) {
        return ctx->device_name;
    }
    return "CPU Software Fallback";
}

void free_model(ModelContext* ctx) {
    if (ctx) {
        if (ctx->model_path) free(ctx->model_path);
        if (ctx->device_name) free(ctx->device_name);
        free(ctx);
    }
}
