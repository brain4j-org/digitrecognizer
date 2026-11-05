package it.echo.digitrecognizer.ai;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.brain4j.core.importing.format.impl.BrainFormat;
import org.brain4j.core.importing.io.ModelIO;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.TrainingConfig;
import org.brain4j.core.training.events.EpochEnd;
import org.brain4j.core.training.events.EpochStart;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.loss.impl.CrossEntropy;
import org.brain4j.math.tensor.Tensor;

import it.echo.digitrecognizer.core.DigitPreprocessor;

public final class MnistTrainer {

    private static final Path DEFAULT_OUTPUT = Path.of("src/main/resources/models/mnist-cnn.zip");

    private MnistTrainer() {
    }

    public static void main(String[] args) throws Exception {
        int epochs = intArg(args, "--epochs", 30);
        int batchSize = intArg(args, "--batch", 128);
        int limit = intArg(args, "--limit", 0);
        double learningRate = doubleArg(args, "--lr", 0.001);
        double minLearningRateFactor = doubleArg(args, "--min-lr-factor", 0.02);
        double dropout = doubleArg(args, "--dropout", 0.1);
        double weightDecay = doubleArg(args, "--weight-decay", 0.001);
        boolean augment = boolArg(args, "--augment", true);
        boolean preprocess = boolArg(args, "--preprocess", false);
        boolean large = boolArg(args, "--large", false);
        boolean wide = boolArg(args, "--wide", false);
        boolean dense2 = boolArg(args, "--dense2", false);
        boolean mild = boolArg(args, "--mild", false);
        double augmentProbability = doubleArg(args, "--augment-probability", 1.0);
        long seed = (long) doubleArg(args, "--seed", 42);
        int evaluateEvery = intArg(args, "--eval-every", 0);
        int swaEpochs = Math.min(Math.max(0, intArg(args, "--swa-epochs", 0)), epochs);
        boolean tta = boolArg(args, "--tta", false);
        String resume = stringArg(args, "--resume", "");
        Path output = Path.of(stringArg(args, "--out", DEFAULT_OUTPUT.toString()));

        System.out.println("Loading MNIST...");
        long started = System.nanoTime();

        MnistLoader.Splits splits = MnistLoader.load();
        List<Sample> trainSamples = take(splits.train(), limit);

        if (preprocess) {
            trainSamples = preprocess(trainSamples);
        }

        List<Sample> testSamples = preprocess ? preprocess(splits.test()) : splits.test();
        ListDataSource test = new ListDataSource(testSamples, false, batchSize);

        if (boolArg(args, "--eval-only", false)) {
            evaluateOnly(args, test, tta);
            return;
        }

        ListDataSource train = augment
            ? new AugmentedDataSource(trainSamples, batchSize, augmenter(seed, mild),
                augmentProbability, seed)
            : new ListDataSource(trainSamples, true, batchSize);

        System.out.printf(Locale.ROOT, "MNIST ready in %.1fs - train=%d, test=%d, batch=%d, augment=%s%n",
            secondsSince(started), train.size(), test.size(), batchSize, augment);

        Sequential model = resume.isBlank()
            ? createModel(dropout, large, wide, dense2)
            : ModelIO.load(new BrainFormat(), Path.of(resume).toFile());

        model.summary();

        AdjustableAdamW optimizer = new AdjustableAdamW(learningRate, weightDecay);
        WeightAverager averager = new WeightAverager();
        Monitor monitor = monitor(model, test, optimizer, learningRate, minLearningRateFactor,
            epochs, evaluateEvery, swaEpochs, averager);
        Trainer trainer = Trainer.of(model, TrainingConfig.of(new CrossEntropy(), optimizer), List.of(monitor));

        long trainingStarted = System.nanoTime();
        trainer.fit(train, epochs);

        System.out.printf(Locale.ROOT, "%nTraining finished in %.1fs%n", secondsSince(trainingStarted));

        EvaluationResult evaluation = model.evaluate(test, new CrossEntropy());
        System.out.println(evaluation.results());

        if (tta) {
            double accuracy = MnistTtaEvaluator.accuracy(model, test.samples());
            System.out.printf(Locale.ROOT, "TTA accuracy: %.2f%%%n", accuracy * 100);
        }

        ModelIO.save(model, new BrainFormat(), output.toFile());
        System.out.printf(Locale.ROOT, "Model saved to %s (%.1f KB)%n",
            output, Files.size(output) / 1024.0);

        Sequential check = ModelIO.load(new BrainFormat(), output.toFile());
        EvaluationResult reloaded = check.evaluate(test, new CrossEntropy());
        System.out.printf(Locale.ROOT, "Reloaded model accuracy: %.2f%%%n", reloaded.accuracy() * 100);

        if (averager.snapshots() > 0) {
            averager.applyTo(model);

            EvaluationResult swaEvaluation = model.evaluate(test, new CrossEntropy());
            System.out.printf(Locale.ROOT, "SWA accuracy (%d snapshots): %.2f%%%n",
                averager.snapshots(), swaEvaluation.accuracy() * 100);

            Path swaOutput = sibling(output, "-swa");
            ModelIO.save(model, new BrainFormat(), swaOutput.toFile());
            System.out.printf(Locale.ROOT, "SWA model saved to %s%n", swaOutput);
        }
    }

    private static Monitor monitor(
        Sequential model,
        ListDataSource test,
        AdjustableAdamW optimizer,
        double learningRate,
        double minLearningRateFactor,
        int epochs,
        int evaluateEvery,
        int swaEpochs,
        WeightAverager averager
    ) {
        return (event, trainer) -> {
            if (event instanceof EpochStart start) {
                optimizer.setLearningRate(
                    scheduledLearningRate(learningRate, minLearningRateFactor, start.epoch(), epochs));
                System.out.printf(Locale.ROOT, "Learning rate: %.6f%n", optimizer.learningRate());
            }

            if (event instanceof EpochEnd end) {
                if (evaluateEvery > 0 && (end.epoch() + 1) % evaluateEvery == 0) {
                    EvaluationResult result = model.evaluate(test, new CrossEntropy());
                    System.out.printf(Locale.ROOT, "Epoch %d test accuracy: %.2f%%%n",
                        end.epoch() + 1, result.accuracy() * 100);
                }

                if (swaEpochs > 0 && end.epoch() >= epochs - swaEpochs) {
                    averager.add(model);
                }
            }
        };
    }

    private static double scheduledLearningRate(
        double learningRate,
        double minFactor,
        int epoch,
        int epochs
    ) {
        if (epochs <= 1) {
            return learningRate;
        }

        double progress = epoch / (double) (epochs - 1);
        double cosine = 0.5 * (1.0 + Math.cos(Math.PI * progress));

        return learningRate * (minFactor + (1.0 - minFactor) * cosine);
    }

    private static void evaluateOnly(String[] args, ListDataSource test, boolean tta) {
        Path modelPath = Path.of(stringArg(args, "--model", DEFAULT_OUTPUT.toString()));

        Sequential model = ModelIO.load(new BrainFormat(), modelPath.toFile());
        EvaluationResult result = model.evaluate(test, new CrossEntropy());

        System.out.println(result.results());
        System.out.printf(Locale.ROOT, "Accuracy: %.2f%%%n", result.accuracy() * 100);

        if (tta) {
            double accuracy = MnistTtaEvaluator.accuracy(model, test.samples());
            System.out.printf(Locale.ROOT, "TTA accuracy: %.2f%%%n", accuracy * 100);
        }
    }

    private static List<Sample> take(List<Sample> samples, int limit) {
        if (limit > 0) {
            return new ArrayList<>(samples.subList(0, Math.min(limit, samples.size())));
        }

        return new ArrayList<>(samples);
    }

    private static List<Sample> preprocess(List<Sample> samples) {
        List<Sample> result = new ArrayList<>(samples.size());

        for (Sample sample : samples) {
            float[] prepared = DigitPreprocessor.prepare(sample.input(0).data());
            result.add(new Sample(
                new Tensor[] {tensor(prepared)},
                sample.labels()));
        }

        return result;
    }

    private static Sequential createModel(double dropout, boolean large, boolean wide, boolean dense2) {
        if (large) {
            return MnistCnn.createLarge(dropout);
        }

        if (wide) {
            return MnistCnn.createWide(dropout);
        }

        if (dense2) {
            return MnistCnn.createDense2(dropout);
        }

        return MnistCnn.create(dropout);
    }

    private static MnistAugmenter augmenter(long seed, boolean mild) {
        if (mild) {
            return new MnistAugmenter(seed, 8.0, 0.08, 0.05, 1.0, 0.2);
        }

        return new MnistAugmenter(seed);
    }

    private static Tensor tensor(float[] pixels) {
        return Tensors.create(new int[] {1, MnistCnn.IMAGE_SIZE, MnistCnn.IMAGE_SIZE}, pixels);
    }

    private static Path sibling(Path path, String suffix) {
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');

        if (dot <= 0) {
            return path.resolveSibling(name + suffix);
        }

        return path.resolveSibling(name.substring(0, dot) + suffix + name.substring(dot));
    }

    private static double secondsSince(long started) {
        return (System.nanoTime() - started) / 1_000_000_000.0;
    }

    private static String stringArg(String[] args, String name, String fallback) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                return args[i + 1];
            }
        }

        return fallback;
    }

    private static int intArg(String[] args, String name, int fallback) {
        return Integer.parseInt(stringArg(args, name, String.valueOf(fallback)));
    }

    private static double doubleArg(String[] args, String name, double fallback) {
        return Double.parseDouble(stringArg(args, name, String.valueOf(fallback)));
    }

    private static boolean boolArg(String[] args, String name, boolean fallback) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                return Boolean.parseBoolean(args[i + 1]);
            }
        }

        return fallback;
    }
}
