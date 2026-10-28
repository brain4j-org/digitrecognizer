package it.echo.digitrecognizer.ai;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.brain4j.core.importing.format.impl.BrainFormat;
import org.brain4j.core.importing.io.ModelIO;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.TrainingConfig;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.datasets.Datasets;
import org.brain4j.datasets.api.Dataset;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.loss.impl.CrossEntropy;

public final class MnistTrainer {

    private static final Path DEFAULT_OUTPUT = Path.of("src/main/resources/models/mnist-cnn.zip");

    private MnistTrainer() {
    }

    public static void main(String[] args) throws Exception {
        int epochs = intArg(args, "--epochs", 10);
        int batchSize = intArg(args, "--batch", 128);
        int limit = intArg(args, "--limit", 0);
        double split = doubleArg(args, "--split", 0.9);
        double learningRate = doubleArg(args, "--lr", 0.0001);
        Path output = Path.of(stringArg(args, "--out", DEFAULT_OUTPUT.toString()));

        System.out.println("Loading MNIST (split " + split + ")...");
        long started = System.nanoTime();

        Dataset dataset = Datasets.mnist(true, batchSize, split);
        ListDataSource train = reshape(dataset.train(), batchSize, true);
        ListDataSource test = reshape(dataset.test(), batchSize, false);

        if (limit > 0) {
            train = subset(train, limit, batchSize);
        }

        System.out.printf(Locale.ROOT, "MNIST ready in %.1fs - train=%d, test=%d, batch=%d%n",
            secondsSince(started), train.size(), test.size(), batchSize);

        Sequential model = MnistCnn.create();
        model.summary();

        Trainer trainer = Trainer.of(model, TrainingConfig.of(new CrossEntropy(), new AdamW(learningRate)));
        long trainingStarted = System.nanoTime();
        trainer.fit(train, epochs);

        System.out.printf(Locale.ROOT, "%nTraining finished in %.1fs%n", secondsSince(trainingStarted));

        EvaluationResult trainEvaluation = model.evaluate(train, new CrossEntropy());
        System.out.printf(Locale.ROOT, "Train accuracy: %.2f%% (loss %.4f)%n",
            trainEvaluation.accuracy() * 100, trainEvaluation.loss());

        EvaluationResult evaluation = model.evaluate(test, new CrossEntropy());
        System.out.println(evaluation.results());

        ModelIO.save(model, new BrainFormat(), output.toFile());
        System.out.printf(Locale.ROOT, "Model saved to %s (%.1f KB)%n",
            output, java.nio.file.Files.size(output) / 1024.0);

        Sequential check = ModelIO.load(new BrainFormat(), output.toFile());
        EvaluationResult reloaded = check.evaluate(test, new CrossEntropy());
        System.out.printf(Locale.ROOT, "Reloaded model accuracy: %.2f%%%n", reloaded.accuracy() * 100);
    }

    private static ListDataSource reshape(ListDataSource source, int batchSize, boolean shuffle) {
        List<Sample> samples = new ArrayList<>(source.size());

        for (Sample sample : source.samples()) {
            samples.add(new Sample(
                new Tensor[] {sample.input(0).reshape(1, MnistCnn.IMAGE_SIZE, MnistCnn.IMAGE_SIZE)},
                sample.labels()));
        }

        return new ListDataSource(samples, shuffle, batchSize);
    }

    private static ListDataSource subset(ListDataSource source, int limit, int batchSize) {
        int size = Math.min(limit, source.size());
        return new ListDataSource(new ArrayList<>(source.samples().subList(0, size)), true, batchSize);
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
}
