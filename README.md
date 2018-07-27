# Intel benchmarks

## How to run benchmarks

### MNIST

MNIST benchmarks are located in `mnist/` directory.
There are 2 different benchmarks: Deep MNIST and MNIST with estimator usage.

To run deep MNIST:

```bash
python mnist/bench_mnist.py --data-dir data --iterations 50000 --batch-size 50
```

To run MNIST with estimator:

```
python mnist/tf_mnist_estimator.py --data-dir data --iterations 50000 --batch-size 50
```

By default, it automatically downloads MNIST dataset in the specified directory (if it does not exist).
At the end of execution, it will write benchmark result, something like:
```
--------------------------------------------------
Benchmark result:
10000 ops, (757.016 op/s)
Total time: 13.210s
--------------------------------------------------
```

### Speech

There are `.lua` script file for `wrk` utility and test data for speech2text model in `speech` directory.
To make this running, need to download and install `wrk`: https://github.com/wg/wrk.
`wrk` is a modern tool for benchmarking HTTP connections and requests.

Also, it requires `speech2text` model, `tfservable-proxy` and `ml-serving` (aka `kuberlab-serving`) for
running an inference model server and proxy it through HTTP.

### Styles

Directory `styles` contains image example used for inference and some test scripts to use with `wrk` (as
for **speech** described above). It also requires `styles-transfer` model, `tfservable-proxy` and `ml-serving`
for running this test.

### TF-servable proxy

TF-servable proxy can be found at https://github.com/kuberlab/tfservable-proxy.
