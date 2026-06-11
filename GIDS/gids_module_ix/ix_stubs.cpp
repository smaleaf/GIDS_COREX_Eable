// Stub library for missing Corex 4.5.0 symbols
// The PyTorch 4.5.0 wheel references these CUPTI symbols that don't exist in 4.3.0 driver.
// They are profiling-only, not needed for training.

extern "C" {

int ixptiActivityEnable(unsigned int kind) { return 0; }
int ixptiActivityDisable(unsigned int kind) { return 0; }
int ixptiActivityRegisterCallbacks(void* cb, void* fn) { return 0; }
int ixptiActivityFlushAll(unsigned int flag) { return 0; }
int ixptiActivityGetNextRecord(unsigned char* buf, size_t sz, void** rec) { return 0; }
int ixptiActivityGetNumDroppedRecords(void* ctx, unsigned int stream, size_t* n) { return 0; }
int ixptiGetTimestamp() { return 0; }
int ixptiGetDeviceCaps(int dev, int* major, int* minor) { return 0; }
int ixptiGetDeviceProperties(int dev, void* props) { return 0; }
int ixptiEnableCallback(int flag, void* subs, void* cb, void* ud) { return 0; }
int ixptiDisableCallback(int flag, void* subs, void* cb) { return 0; }
int ixptiSubscribe(void** subs, void* cb, void* ud) { return 0; }
int ixptiUnsubscribe(void* subs) { return 0; }
int ixptiEventCategoryGetInfo(void* ev, void* info) { return 0; }
int ixptiEventGetAttribute(void* ev, size_t sz, void* attr, size_t* n) { return 0; }
int ixptiEventGetId(void* ev, unsigned int* id) { return 0; }

// Some CUDA 11.x-style symbols that may be missing in 10.2
int cudaDeviceGetAttribute(void* val, int attr, int dev) { return 0; }
int cudaFuncGetAttributes(void* attr, const void* fn) { return 0; }
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, const void* fn, int bs, size_t sm) { return 0; }
int cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* n, const void* fn, int bs, size_t sm, unsigned int f) { return 0; }

}