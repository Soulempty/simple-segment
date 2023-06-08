import os
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger()

class TRT(object):
    def __init__(self,engine_file,input_size=[],binding_names=['input','output']):
        super().__init__()
        assert os.path.exists(engine_file)
        self.engine_file = engine_file
        self.input_size =input_size
        self.binding_names = binding_names
        self.engine = None
        self.context = None
        self.stream = cuda.Stream()
        self.buffers = []
        self.outputs = []
        self.bindings = []
        self.create_context()

    def create_context(self):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_file, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(self.engine.get_binding_index("input"), (1, 3, *self.input_size))
        for i,binding_name in enumerate(self.binding_names):
            binding_idx = self.engine.get_binding_index(binding_name)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_name))
            occupy = cuda.pagelocked_empty(size, dtype)
            buffer = cuda.mem_alloc(occupy.nbytes)
            self.buffers.append(buffer)
            self.bindings.append(int(buffer))
            if i>0:
                self.outputs.append(occupy)

    def infer(self,input):
        input_buffer = np.ascontiguousarray(input)
        cuda.memcpy_htod_async(self.buffers[0], input_buffer, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for i in range(1,len(self.buffers)):
            cuda.memcpy_dtoh_async(self.outputs[i-1], self.buffers[i], self.stream)
        self.stream.synchronize()
        return self.outputs