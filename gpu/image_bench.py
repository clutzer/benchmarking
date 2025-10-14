import argparse
import os
import time
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='val2017', help='path to dataset directory')
parser.add_argument('--data-size', type=int, default=16, help='amount of images to load into memory; must be >= batch-size; bigger = more memory')
parser.add_argument('--batch-size', type=int, default=1, help='batch size for inference; default 1')
parser.add_argument('--num-workers', type=int, default=1, help='number of workers; default 1')
parser.add_argument('--multiprocessing', action='store_true', help='use multiprocessing instead of multithreading')
parser.add_argument('--framework', default='pytorch', help='pytorch (default) or tensorflow')
parser.add_argument('--model', type=int, default=0, help='inference model (0=mobilenet (default), 1=efficientnet, 2=resnet)')
parser.add_argument('--duration', type=int, default=120, help='number of seconds to run benchmark; default 120s')
parser.add_argument('--pytorch-threads', type=int, default=0, help='number of threads to allocate to pytorch; if 0 (default), pytorch auto assigns based on core count')
args = parser.parse_args()

# import the "right" symbols for either multiprocessing or multithreading
if args.multiprocessing:
    from multiprocessing import Process as Worker, Lock, Array
else:
    from threading import Thread as Worker, Lock
    def Array(_, x): return x

# PyTorch Benchmark Driver
import torch
from torchvision import models, transforms
class BenchPyTorch:
    # static class members are shared when multithreading and duplicated when multiprocessing
    # this means 1 model shared by n-threads for multithreading and 1 model per process when multiprocessing
    mutex = Lock()
    model = None
    device = None
    def get_model(model):        
        with BenchPyTorch.mutex:
            if not BenchPyTorch.model:
                if args.pytorch_threads:
                    torch.set_num_threads(args.pytorch_threads)
                BenchPyTorch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if model == 0:
                    model = models.mobilenet_v2(weights='DEFAULT')
                if model == 1:
                    model = models.efficientnet_v2_s(weights='DEFAULT')
                if model == 2:
                    model = models.resnet152(weights='DEFAULT')
                BenchPyTorch.model = model.to(BenchPyTorch.device)
                BenchPyTorch.model.eval()
        return BenchPyTorch.model, BenchPyTorch.device

    def __init__(self, model):
        self.model, self.device = BenchPyTorch.get_model(model)

    data = None
    def load_data(self, imgs, batch_size):
        with BenchPyTorch.mutex:
            if not BenchPyTorch.data:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                data = [transform(img) for img in imgs]
                data = torch.stack(data)
                BenchPyTorch.data = data.split(batch_size)

    def run(self, i):
        t0 = time.time()
        batch_t = BenchPyTorch.data[i % len(BenchPyTorch.data)].to(self.device)
        with torch.no_grad():
            t1 = time.time()
            bat_latency = 1000 * (t1 - t0)
            result = self.model(batch_t)
            inf_latency = 1000 * (time.time() - t1)
        return result, bat_latency, inf_latency

# Tensorflow Benchmark Driver
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet152V2
class BenchTensorflow:
    mutex = Lock()
    model = None
    preprocess = None
    def get_model(model):
        with BenchTensorflow.mutex:
            if not BenchTensorflow.model:
                if model == 0:
                    BenchTensorflow.model = MobileNetV2(weights='imagenet')
                    BenchTensorflow.preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
                if model == 1:
                    BenchTensorflow.model = EfficientNetB0(weights='imagenet')
                    BenchTensorflow.preprocess = tf.keras.applications.efficientnetb0.preprocess_input
                if model == 2:
                    BenchTensorflow.model = ResNet152V2(weights='imagenet')
                    BenchTensorflow.preprocess = tf.keras.applications.resnet152v2.preprocess_input
        return BenchTensorflow.model, BenchTensorflow.preprocess

    def __init__(self, model):
        self.model, self.preprocess = BenchTensorflow.get_model(model)

    data = None
    def load_data(self, imgs, batch_size):
        with BenchTensorflow.mutex:
            if not BenchTensorflow.data:
                data = np.array([np.array(img.resize((224, 224))) for img in imgs])
                data = self.preprocess(data)
                BenchTensorflow.data = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
        self.iter = iter(BenchTensorflow.data)

    def run(self, i):
        t0 = time.time()
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(BenchTensorflow.data)
            batch = next(self.iter)
        t1 = time.time()
        bat_latency = 1000 * (t1 - t0)
        result = self.model.predict(batch, verbose=0)
        inf_latency = 1000 * (time.time() - t1)
        return result, bat_latency, inf_latency

def worker(n, mutex, state):
    assert args.framework in ('pytorch', 'tensorflow')
    if args.framework == 'pytorch':
        bench = BenchPyTorch(args.model)
    if args.framework == 'tensorflow':
        bench = BenchTensorflow(args.model)

    imgs = [Image.open(os.path.join(args.dataset, f)).convert('RGB') for f in os.listdir(args.dataset)[:args.data_size]]
    bench.load_data(imgs, args.batch_size)

    # prewarm the model
    for i in range(4):
        bench.run(i)

    with mutex:
        state[0] += 1

    while state[0] < 0:
        pass

    i = 0
    while state[0] >= 0:
        result, bat_latency, inf_latency = bench.run(i)
        with mutex:
            state[1] += 1 # batches
            state[2] += len(result) # inferences
            state[3] += bat_latency # total latency
            state[4] = min(state[4], bat_latency) # min latency
            state[5] = max(state[5], bat_latency) # max latency
            state[6] += inf_latency # total latency
            state[7] = min(state[7], inf_latency) # min latency
            state[8] = max(state[8], inf_latency) # max latency
        i += 1

def main():
    mutex = Lock()

    # global state for tracking the metrics, fields are:
    # - worker synchronization flag: <0 = wait to start; then >=0 = run; then <0 again = stop and exit worker
    # - total number of inferences completed
    # - total cumulative batching latency (ms)
    # - min batching latency (ms)
    # - max batching latency (ms)
    # - total cumulative inference latency (ms)
    # - min inference latency (ms)
    # - max inference latency (ms)
    state = Array('f', [-args.num_workers, 0, 0, 0, 99999, 0, 0, 99999, 0])

    workers = []
    for i in range(args.num_workers):
        w = Worker(target=worker, args=(i, mutex, state))
        w.start()
        workers.append(w)

    # wait for workers to start
    while state[0] < 0:
        pass

    # run benchmark
    t = time.time()
    now = time.time()
    while now - t <= args.duration:
        time.sleep(0.1)
        now = time.time()
        with mutex:
            print(f'\rmodel={args.model} workers={args.num_workers} batch={args.batch_size:4d}, {now - t:6.2f}s elapsed, {int(state[2]):6d} imgs processed, {state[2]/(now - t):7.2f} imgs/s {state[4]:7.1f}/{state[3]/state[1] if state[1] else 0:7.1f}/{state[5]:7.1f}ms bat latency {state[7]:7.1f}/{state[6]/state[1] if state[1] else 0:7.1f}/{state[8]:7.1f}ms inf latency', end='')
    state[0] = -1 # stop
    print()
    for w in workers:
        w.join()

if __name__ == '__main__':
    main()
