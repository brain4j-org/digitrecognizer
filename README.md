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

## GPU note

The `Use GPU` checkbox creates a Silicon/CUDA device (or Metal on macOS) and moves the model parameters to the GPU. The first use compiles the kernels through Slang and can take a few seconds. If `slangc` is not installed, the app shows `GPU unavailable: Slang not found` and continues using the CPU. You can provide an explicit path with the `SLANGC` environment variable or the `-Dsilicon.slangc=/path/to/slangc` property.

On Linux the CUDA backend also requires NVRTC. If `slangc` reports `failed to load downstream compiler 'nvrtc'`, install the NVRTC libraries; on Fedora 44:

```bash
sudo dnf install cuda-nvrtc-13-2 cuda-nvrtc-devel-13-2
```

The `cuda-nvrtc-devel` package provides the `libnvrtc.so` symlink used by the Slang loader.
