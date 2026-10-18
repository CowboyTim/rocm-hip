// inference_log.h -- diagnostics of the inference library go to stderr only when INFERENCE_VERBOSE is set
// (to anything but "0"): a library must not write to a host program's stderr on its own. Errors are reported
// through return codes and inference_get_last_error.

#ifndef INFERENCE_LOG_H
#define INFERENCE_LOG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int inference_verbose(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("INFERENCE_VERBOSE");
        v = e && e[0] && strcmp(e, "0") != 0;
    }
    return v;
}

#define INFER_LOG(...)                                   \
    do {                                                 \
        if (inference_verbose()) fprintf(stderr, __VA_ARGS__); \
    } while (0)

#endif // INFERENCE_LOG_H
