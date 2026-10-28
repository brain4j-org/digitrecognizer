package it.echo.digitrecognizer;

import java.awt.BorderLayout;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;

import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import javax.swing.Timer;
import javax.swing.border.EmptyBorder;

import it.echo.digitrecognizer.ai.DigitRecognizer;
import it.echo.digitrecognizer.core.DigitPreprocessor;
import it.echo.digitrecognizer.ui.Card;
import it.echo.digitrecognizer.ui.DigitCanvas;
import it.echo.digitrecognizer.ui.InputPreview;
import it.echo.digitrecognizer.ui.PredictionPanel;
import it.echo.digitrecognizer.ui.ProbabilityPanel;
import it.echo.digitrecognizer.ui.Theme;

public final class MainFrame extends JFrame {

    private final DigitCanvas canvas = new DigitCanvas();
    private final PredictionPanel predictionPanel = new PredictionPanel();
    private final ProbabilityPanel probabilityPanel = new ProbabilityPanel();
    private final InputPreview inputPreview = new InputPreview();
    private final JLabel statusLabel = new JLabel("Loading model…");
    private final JButton undoButton = createButton("Undo", false);
    private final JButton clearButton = createButton("Clear", true);
    private final JCheckBox gpuCheckBox = new JCheckBox("Use GPU");

    private final Timer debounce;
    private final AtomicLong requestId = new AtomicLong();

    private DigitRecognizer recognizer;
    private volatile boolean modelReady;
    private boolean ignoreGpuToggle;
    private boolean switchingDevice;

    public MainFrame() {
        super("Digit Recognizer");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setMinimumSize(new Dimension(1000, 740));
        setSize(1200, 820);
        setLocationRelativeTo(null);

        debounce = new Timer(120, event -> runInference());
        debounce.setRepeats(false);

        canvas.setListener(this::onCanvasChanged);
        undoButton.setEnabled(false);
        undoButton.addActionListener(event -> canvas.undo());
        clearButton.addActionListener(event -> canvas.clear());

        setContentPane(buildRoot());

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent event) {
                debounce.stop();
                if (recognizer != null) {
                    recognizer.close();
                }
            }
        });

        installShortcuts();
        loadModel();
    }

    private JPanel buildRoot() {
        JPanel root = new JPanel(new BorderLayout(0, 18));
        root.setBackground(Theme.BG);
        root.setBorder(new EmptyBorder(24, 30, 20, 30));

        root.add(buildHeader(), BorderLayout.NORTH);
        root.add(buildContent(), BorderLayout.CENTER);

        return root;
    }

    private JComponent buildHeader() {
        JPanel header = new JPanel(new BorderLayout(20, 0));
        header.setOpaque(false);
        header.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createMatteBorder(0, 0, 1, 0, Theme.LINE),
            new EmptyBorder(0, 0, 15, 0)));

        JPanel titles = new JPanel();
        titles.setOpaque(false);
        titles.setLayout(new BoxLayout(titles, BoxLayout.Y_AXIS));

        JPanel wordmark = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        wordmark.setOpaque(false);
        wordmark.setAlignmentX(LEFT_ALIGNMENT);

        JLabel first = new JLabel("Digit");
        first.setFont(Theme.font(Font.BOLD, 28f));
        first.setForeground(Theme.INK);

        JLabel slash = new JLabel("/");
        slash.setFont(Theme.font(Font.BOLD, 28f));
        slash.setForeground(Theme.ACCENT);

        JLabel second = new JLabel("Recognizer");
        second.setFont(Theme.font(Font.BOLD, 28f));
        second.setForeground(Theme.INK);

        wordmark.add(first);
        wordmark.add(slash);
        wordmark.add(second);

        titles.add(wordmark);

        JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 0, 0));
        right.setOpaque(false);

        gpuCheckBox.setOpaque(false);
        gpuCheckBox.setFont(Theme.mono(Font.PLAIN, 10.5f));
        gpuCheckBox.setForeground(Theme.BONE_SOFT);
        gpuCheckBox.setFocusPainted(false);
        gpuCheckBox.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        gpuCheckBox.setToolTipText("Load the model on the GPU and use hardware acceleration");
        gpuCheckBox.setEnabled(false);
        gpuCheckBox.addActionListener(event -> toggleGpu());
        right.add(gpuCheckBox);

        header.add(titles, BorderLayout.WEST);
        header.add(right, BorderLayout.EAST);
        return header;
    }

    private JComponent buildContent() {
        JPanel content = new JPanel(new BorderLayout(20, 0));
        content.setOpaque(false);
        content.add(buildCanvasCard(), BorderLayout.CENTER);
        content.add(buildSidePanel(), BorderLayout.EAST);
        return content;
    }

    private JComponent buildCanvasCard() {
        Card card = new Card(2);
        card.setBorder(new EmptyBorder(18, 18, 14, 18));
        card.setLayout(new BorderLayout(0, 14));

        JPanel header = new JPanel(new BorderLayout(12, 0));
        header.setOpaque(false);

        JPanel labels = new JPanel();
        labels.setOpaque(false);
        labels.setLayout(new BoxLayout(labels, BoxLayout.Y_AXIS));

        JLabel title = new JLabel("Canvas");
        title.setFont(Theme.font(Font.BOLD, 16f));
        title.setForeground(Theme.INK);

        JLabel hint = new JLabel("28 × 28");
        hint.setFont(Theme.mono(Font.PLAIN, 10.5f));
        hint.setForeground(Theme.FAINT);

        labels.add(title);
        labels.add(Box.createVerticalStrut(3));
        labels.add(hint);

        JPanel brush = new JPanel();
        brush.setOpaque(false);
        brush.setLayout(new BoxLayout(brush, BoxLayout.X_AXIS));

        JLabel brushLabel = new JLabel("/ Brush size");
        brushLabel.setFont(Theme.mono(Font.PLAIN, 10.5f));
        brushLabel.setForeground(Theme.MUTED);

        JSlider slider = new JSlider(10, 36, 20);
        slider.setOpaque(false);
        slider.setPreferredSize(new Dimension(112, 28));
        slider.setMaximumSize(new Dimension(112, 28));
        slider.setToolTipText("Brush size");
        slider.addChangeListener(event -> canvas.setBrushCells(slider.getValue() / 10f));

        brush.add(brushLabel);
        brush.add(Box.createHorizontalStrut(10));
        brush.add(slider);

        header.add(labels, BorderLayout.WEST);
        header.add(brush, BorderLayout.EAST);

        JPanel footer = new JPanel(new BorderLayout(16, 0));
        footer.setOpaque(false);
        footer.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createMatteBorder(1, 0, 0, 0, Theme.BORDER_SOFT),
            new EmptyBorder(12, 0, 0, 0)));

        JPanel actions = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        actions.setOpaque(false);
        actions.add(undoButton);
        actions.add(clearButton);

        JLabel shortcutHint = new JLabel("Ctrl+Z · C");
        shortcutHint.setFont(Theme.mono(Font.PLAIN, 9.5f));
        shortcutHint.setForeground(Theme.alpha(Theme.FAINT, 180));
        shortcutHint.setBorder(new EmptyBorder(0, 8, 0, 0));
        actions.add(shortcutHint);

        JPanel preview = new JPanel();
        preview.setOpaque(false);
        preview.setLayout(new BoxLayout(preview, BoxLayout.Y_AXIS));

        JLabel previewLabel = new JLabel("/ Input");
        previewLabel.setFont(Theme.mono(Font.PLAIN, 9.5f));
        previewLabel.setForeground(Theme.FAINT);
        previewLabel.setAlignmentX(0.5f);

        inputPreview.setPreferredSize(new Dimension(44, 44));
        inputPreview.setMaximumSize(new Dimension(44, 44));
        inputPreview.setAlignmentX(0.5f);

        preview.add(previewLabel);
        preview.add(Box.createVerticalStrut(4));
        preview.add(inputPreview);

        footer.add(actions, BorderLayout.WEST);
        footer.add(preview, BorderLayout.EAST);

        card.add(header, BorderLayout.NORTH);
        card.add(canvas, BorderLayout.CENTER);
        card.add(footer, BorderLayout.SOUTH);

        return card;
    }

    private JComponent buildSidePanel() {
        JPanel side = new JPanel();
        side.setOpaque(false);
        side.setLayout(new BoxLayout(side, BoxLayout.Y_AXIS));
        side.setPreferredSize(new Dimension(368, 0));

        Card predictionCard = new Card(2);
        predictionCard.setLayout(new BorderLayout());
        predictionCard.setAlignmentX(LEFT_ALIGNMENT);
        predictionCard.setPreferredSize(new Dimension(368, 190));
        predictionCard.setMaximumSize(new Dimension(Integer.MAX_VALUE, 190));
        predictionCard.setMinimumSize(new Dimension(0, 190));
        predictionCard.add(predictionPanel, BorderLayout.CENTER);

        Card probabilitiesCard = new Card(2);
        probabilitiesCard.setLayout(new BorderLayout());
        probabilitiesCard.setAlignmentX(LEFT_ALIGNMENT);
        probabilitiesCard.setPreferredSize(new Dimension(368, 410));
        probabilitiesCard.setMaximumSize(new Dimension(Integer.MAX_VALUE, 500));
        probabilitiesCard.setMinimumSize(new Dimension(0, 300));
        probabilitiesCard.add(probabilityPanel, BorderLayout.CENTER);

        JPanel footer = new JPanel();
        footer.setOpaque(false);
        footer.setAlignmentX(LEFT_ALIGNMENT);
        footer.setLayout(new BoxLayout(footer, BoxLayout.Y_AXIS));
        footer.setBorder(new EmptyBorder(0, 4, 0, 4));
        footer.setMaximumSize(new Dimension(Integer.MAX_VALUE, 42));
        footer.setPreferredSize(new Dimension(368, 42));

        statusLabel.setFont(Theme.mono(Font.PLAIN, 13.5f));
        statusLabel.setForeground(Theme.MUTED);

        footer.add(statusLabel);
        footer.add(Box.createVerticalStrut(3));

        side.add(predictionCard);
        side.add(Box.createVerticalStrut(12));
        side.add(probabilitiesCard);
        side.add(Box.createVerticalGlue());
        side.add(footer);

        return side;
    }

    private JButton createButton(String text, boolean primary) {
        JButton button = new JButton(text);
        button.setFont(Theme.mono(Font.BOLD, 10.5f));
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        button.setMargin(new Insets(7, 15, 7, 15));
        button.setPreferredSize(new Dimension(92, 34));

        if (primary) {
            button.setBackground(Theme.INK);
            button.setForeground(Theme.PAPER_LIGHT);
            button.putClientProperty("JButton.borderColor", Theme.INK);
            button.putClientProperty("JButton.hoverBackground", Theme.ACCENT);
            button.putClientProperty("JButton.pressedBackground", Theme.INK_SOFT);
        } else {
            button.setBackground(Theme.PAPER_LIGHT);
            button.setForeground(Theme.INK);
            button.putClientProperty("JButton.borderColor", Theme.LINE);
            button.putClientProperty("JButton.hoverBackground", Theme.PAPER_DEEP);
            button.putClientProperty("JButton.pressedBackground", Theme.LINE_SOFT);
        }

        return button;
    }

    private void toggleGpu() {
        if (ignoreGpuToggle || switchingDevice || recognizer == null || !modelReady) {
            return;
        }

        boolean useGpu = gpuCheckBox.isSelected();
        switchingDevice = true;
        gpuCheckBox.setEnabled(false);
        debounce.stop();
        requestId.incrementAndGet();
        predictionPanel.setBusy(true);
        statusLabel.setForeground(Theme.MUTED);
        statusLabel.setText(useGpu ? "Loading model on GPU…" : "Moving model back to CPU…");

        CompletableFuture<Void> switching = useGpu
            ? recognizer.useGpuAsync()
            : recognizer.useCpuAsync();

        switching.whenComplete((ignored, error) -> SwingUtilities.invokeLater(() -> {
            switchingDevice = false;
            gpuCheckBox.setEnabled(true);

            if (error != null) {
                ignoreGpuToggle = true;
                gpuCheckBox.setSelected(!useGpu);
                ignoreGpuToggle = false;

                String detail = rootCause(error).getMessage();
                gpuCheckBox.setToolTipText(detail);
                statusLabel.setForeground(Theme.DANGER);

                if (!useGpu) {
                    statusLabel.setText("Could not switch back to CPU");
                } else if (detail != null && detail.toLowerCase(Locale.ROOT).contains("slang")) {
                    statusLabel.setText("GPU unavailable: Slang not found");
                } else {
                    statusLabel.setText("GPU unavailable");
                }

                predictionPanel.setBusy(false);
                return;
            }

            gpuCheckBox.setToolTipText(useGpu
                ? "Model loaded on " + recognizer.deviceName()
                : "Load the model on the GPU and use hardware acceleration");
            statusLabel.setForeground(Theme.MUTED);
            statusLabel.setText(useGpu ? "Model on GPU" : "Model on CPU");

            if (canvas.hasInk()) {
                onCanvasChanged(true);
            } else {
                predictionPanel.setBusy(false);
            }
        }));
    }

    private void installShortcuts() {
        JComponent root = getRootPane();
        int menuMask = java.awt.Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();

        root.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW)
            .put(KeyStroke.getKeyStroke(KeyEvent.VK_C, 0), "clearCanvas");
        root.getActionMap().put("clearCanvas", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent event) {
                canvas.clear();
            }
        });

        root.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW)
            .put(KeyStroke.getKeyStroke(KeyEvent.VK_Z, menuMask), "undoCanvas");
        root.getActionMap().put("undoCanvas", new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent event) {
                canvas.undo();
            }
        });
    }

    private void loadModel() {
        statusLabel.setText("Loading model…");

        new SwingWorker<DigitRecognizer, Void>() {
            @Override
            protected DigitRecognizer doInBackground() throws Exception {
                return DigitRecognizer.loadBundled();
            }

            @Override
            protected void done() {
                try {
                    recognizer = get();
                    modelReady = true;
                    gpuCheckBox.setEnabled(true);
                    statusLabel.setForeground(Theme.MUTED);
                    statusLabel.setText("Automatic recognition");

                    if (canvas.hasInk()) {
                        onCanvasChanged(true);
                    }
                } catch (InterruptedException exception) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException exception) {
                    modelReady = false;
                    exception.printStackTrace(System.err);
                    statusLabel.setForeground(Theme.DANGER);
                    statusLabel.setText("Model unavailable");
                }
            }
        }.execute();
    }

    private void onCanvasChanged(boolean hasInk) {
        undoButton.setEnabled(canvas.canUndo());
        debounce.stop();
        requestId.incrementAndGet();

        if (!hasInk) {
            predictionPanel.clear();
            probabilityPanel.clear();
            inputPreview.clear();
            statusLabel.setForeground(Theme.MUTED);
            statusLabel.setText(modelReady ? "Waiting for a stroke" : "Loading model…");
            return;
        }

        if (!modelReady) {
            statusLabel.setForeground(Theme.MUTED);
            statusLabel.setText("Model loading…");
            return;
        }

        predictionPanel.setBusy(true);
        statusLabel.setForeground(Theme.MUTED);
        statusLabel.setText("Recognizing…");
        debounce.restart();
    }

    private void runInference() {
        if (!modelReady || !canvas.hasInk() || recognizer == null) {
            return;
        }

        float[] input = DigitPreprocessor.prepare(canvas.toGrid());
        inputPreview.setInput(input);

        long request = requestId.incrementAndGet();
        long startedAt = System.nanoTime();

        recognizer.predictAsync(input).whenComplete((probabilities, error) ->
            SwingUtilities.invokeLater(() -> {
                if (request != requestId.get()) {
                    return;
                }

                if (error != null) {
                    error.printStackTrace(System.err);
                    predictionPanel.setBusy(false);
                    probabilityPanel.clear();
                    statusLabel.setForeground(Theme.DANGER);
                    statusLabel.setText("Recognition failed");
                    return;
                }

                long elapsedMs = Math.max(1, (System.nanoTime() - startedAt) / 1_000_000);
                predictionPanel.setPredictions(probabilities);
                probabilityPanel.setPredictions(probabilities);
                statusLabel.setForeground(Theme.MUTED);
                statusLabel.setText("Recognized in " + elapsedMs + " ms");
            }));
    }

    private static Throwable rootCause(Throwable throwable) {
        Throwable cause = throwable;
        while (cause.getCause() != null && cause.getCause() != cause) {
            cause = cause.getCause();
        }
        return cause;
    }
}
