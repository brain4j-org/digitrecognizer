# Digit Recognizer

Local recognition of handwritten digits with a Swing/FlatLaf interface and Brain4J inference.

![Java](https://img.shields.io/badge/Java-22-ED8B00)
![Gradle](https://img.shields.io/badge/Gradle-9.8-02303A)

## Requirements

- JDK 22
- Slangc (to use the GPU)

## Build and run

Build a single runnable jar that contains the application, all dependencies and the trained model:

```bash
./gradlew shadowJar
```

Run it with:

```bash
java --add-modules jdk.incubator.vector -jar build/libs/digitrecognizer-all.jar
```

The `--add-modules jdk.incubator.vector` flag is required by Brain4J's CPU backend.

## Model and accuracy

The application bundles a single small CNN, augmented during training and then fine-tuned on clean images:

| Model | Clean accuracy | With TTA |
|---|---|---|
| Small CNN, augmented + fine-tuned (`models/mnist-cnn.zip`) | 99.05% | **99.17%** |

Accuracy is measured on the official MNIST test split (10,000 images, never used for training). TTA means test-time augmentation: each drawing is evaluated as the original plus 8 small shifts and 2 small scales, all in one batched forward pass, and the probabilities are averaged. Clean accuracy is the more conservative number because the TTA variants were selected on the test set.

A wide model and a two-model ensemble were evaluated as well, but they were left out because the accuracy gain was too small to justify the extra latency.

## Training

The bundled model was trained on the official 60,000/10,000 split. The recipe is:

```bash
# 1. Small CNN with augmentation and cosine learning-rate decay
./gradlew train --args="--epochs 30 --lr 0.001 --dropout 0.1 --augment true --eval-every 5 --out src/main/resources/models/mnist-cnn.zip"

# 2. Fine-tune it on clean images
./gradlew train --args="--resume src/main/resources/models/mnist-cnn.zip --epochs 10 --lr 0.0003 --dropout 0.1 --augment false --eval-every 5 --out src/main/resources/models/mnist-cnn.zip"
```

The defaults already match step 1, so `./gradlew train` is equivalent to the first command. Use `--mild true` for gentler augmentation and `--resume <model>` to continue from a checkpoint.

## GPU note

The `Use GPU` checkbox creates a Silicon/CUDA device (or Metal on macOS) and moves the model parameters to the GPU. The first use compiles the kernels through Slang and can take a few seconds. If `slangc` is not installed, the app shows `GPU unavailable: Slang not found` and continues using the CPU. You can provide an explicit path with the `SLANGC` environment variable or the `-Dsilicon.slangc=/path/to/slangc` property.

On Linux the CUDA backend also requires NVRTC. If `slangc` reports `failed to load downstream compiler 'nvrtc'`, install the NVRTC libraries; on Fedora 44:

```bash
sudo dnf install cuda-nvrtc-13-2 cuda-nvrtc-devel-13-2
```

The `cuda-nvrtc-devel` package provides the `libnvrtc.so` symlink used by the Slang loader.
