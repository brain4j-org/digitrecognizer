package it.echo.digitrecognizer.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JComponent;

public final class DigitCanvas extends JComponent {

    public interface Listener {
        void onCanvasChanged(boolean hasInk);
    }

    private static final int GRID = 28;
    private static final int RASTER = GRID * 20;
    private static final double MIN_POINT_DISTANCE = 0.015;

    private record Stroke(List<Point2D.Double> points, float brushCells) {
    }

    private final List<Stroke> strokes = new ArrayList<>();
    private final Font emptyTitleFont = Theme.mono(Font.PLAIN, 13f);
    private final Font emptyHintFont = Theme.mono(Font.PLAIN, 11f);

    private BufferedImage staticLayer;
    private boolean staticLayerDirty = true;
    private int staticLayerWidth;
    private int staticLayerHeight;

    private Listener listener = hasInk -> { };
    private List<Point2D.Double> activePoints;
    private float activeBrush = 2.0f;
    private float brushCells = 2.0f;
    private Point2D.Double cursorCell;
    private boolean drawing;

    public DigitCanvas() {
        setOpaque(true);
        setPreferredSize(new Dimension(460, 460));
        setMinimumSize(new Dimension(300, 300));
        setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
        setFocusable(true);

        MouseAdapter mouse = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent event) {
                requestFocusInWindow();

                Point2D.Double point = toCell(event.getPoint(), false);
                if (point == null) {
                    return;
                }

                activePoints = new ArrayList<>();
                activePoints.add(point);
                activeBrush = brushCells;
                drawing = true;
                repaint();
            }

            @Override
            public void mouseDragged(MouseEvent event) {
                if (!drawing || activePoints == null) {
                    return;
                }

                Point2D.Double point = toCell(event.getPoint(), true);
                if (point == null) {
                    return;
                }

                Point2D.Double last = activePoints.getLast();
                if (last.distance(point) >= MIN_POINT_DISTANCE) {
                    activePoints.add(point);
                    repaint();
                }
            }

            @Override
            public void mouseReleased(MouseEvent event) {
                if (!drawing) {
                    return;
                }

                drawing = false;
                if (activePoints != null && !activePoints.isEmpty()) {
                    Stroke stroke = new Stroke(List.copyOf(activePoints), activeBrush);
                    strokes.add(stroke);
                    appendToStaticLayer(stroke);
                }
                activePoints = null;
                listener.onCanvasChanged(!strokes.isEmpty());
                repaint();
            }

            @Override
            public void mouseMoved(MouseEvent event) {
                cursorCell = toCell(event.getPoint(), false);
                repaint();
            }

            @Override
            public void mouseExited(MouseEvent event) {
                cursorCell = null;
                repaint();
            }
        };

        addMouseListener(mouse);
        addMouseMotionListener(mouse);
    }

    public void setListener(Listener listener) {
        this.listener = listener == null ? hasInk -> { } : listener;
    }

    public void setBrushCells(float cells) {
        brushCells = Math.max(0.8f, Math.min(3.6f, cells));
        repaint();
    }

    public float getBrushCells() {
        return brushCells;
    }

    public boolean hasInk() {
        return !strokes.isEmpty() || activePoints != null;
    }

    public boolean canUndo() {
        return !strokes.isEmpty();
    }

    public void clear() {
        if (strokes.isEmpty() && activePoints == null) {
            return;
        }

        strokes.clear();
        activePoints = null;
        drawing = false;
        staticLayerDirty = true;
        listener.onCanvasChanged(false);
        repaint();
    }

    public void undo() {
        if (strokes.isEmpty()) {
            return;
        }

        strokes.removeLast();
        staticLayerDirty = true;
        listener.onCanvasChanged(!strokes.isEmpty());
        repaint();
    }

    public float[] toGrid() {
        if (!hasInk()) {
            return new float[GRID * GRID];
        }

        BufferedImage raster = new BufferedImage(RASTER, RASTER, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = raster.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);

        for (Stroke stroke : strokes) {
            drawRasterStroke(g, stroke);
        }

        if (activePoints != null && !activePoints.isEmpty()) {
            drawRasterStroke(g, new Stroke(activePoints, activeBrush));
        }

        g.dispose();

        int[] pixels = raster.getRGB(0, 0, RASTER, RASTER, null, 0, RASTER);
        float[] grid = new float[GRID * GRID];
        int block = RASTER / GRID;

        for (int y = 0; y < GRID; y++) {
            for (int x = 0; x < GRID; x++) {
                int sum = 0;
                for (int py = 0; py < block; py++) {
                    int row = (y * block + py) * RASTER + x * block;
                    for (int px = 0; px < block; px++) {
                        sum += (pixels[row + px] >>> 24) & 0xFF;
                    }
                }
                grid[y * GRID + x] = sum / (255f * block * block);
            }
        }

        return grid;
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        Graphics2D g = (Graphics2D) graphics.create();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        Rectangle board = boardBounds();
        if (board.width <= 0) {
            g.dispose();
            return;
        }

        ensureStaticLayer();
        if (staticLayer != null) {
            g.drawImage(staticLayer, 0, 0, null);
        } else {
            g.setColor(Theme.CARD_TOP);
            g.fillRect(0, 0, getWidth(), getHeight());
        }

        RoundRectangle2D shape = boardShape(board);
        g.clip(shape);

        if (activePoints != null && !activePoints.isEmpty()) {
            drawScreenStroke(g, board, new Stroke(activePoints, activeBrush));
        }

        if (cursorCell != null && !drawing) {
            drawCursor(g, board);
        }
        g.setClip(null);

        g.setColor(Theme.LINE);
        g.setStroke(new BasicStroke(1f));
        g.draw(shape);

        if (!hasInk()) {
            drawEmptyState(g, board);
        }

        g.dispose();
    }

    private void ensureStaticLayer() {
        int width = getWidth();
        int height = getHeight();
        if (width <= 0 || height <= 0) {
            return;
        }

        if (staticLayer == null || width != staticLayerWidth || height != staticLayerHeight) {
            staticLayer = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            staticLayerWidth = width;
            staticLayerHeight = height;
            staticLayerDirty = true;
        }

        if (!staticLayerDirty) {
            return;
        }

        Graphics2D g = staticLayer.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Theme.CARD_TOP);
        g.fillRect(0, 0, width, height);

        Rectangle board = boardBounds();
        RoundRectangle2D shape = boardShape(board);

        g.setColor(Theme.CARD_INNER);
        g.fill(shape);
        drawGrid(g, board);
        drawCornerMarks(g, board);

        g.clip(shape);
        for (Stroke stroke : strokes) {
            drawScreenStroke(g, board, stroke);
        }
        g.setClip(null);
        g.dispose();

        staticLayerDirty = false;
    }

    private void appendToStaticLayer(Stroke stroke) {
        if (staticLayer == null
            || staticLayerDirty
            || staticLayerWidth != getWidth()
            || staticLayerHeight != getHeight()) {
            staticLayerDirty = true;
            return;
        }

        Graphics2D g = staticLayer.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        Rectangle board = boardBounds();
        g.clip(boardShape(board));
        drawScreenStroke(g, board, stroke);
        g.dispose();
    }

    private RoundRectangle2D boardShape(Rectangle board) {
        return new RoundRectangle2D.Float(
            board.x + 0.5f, board.y + 0.5f, board.width - 1f, board.height - 1f, 3, 3);
    }

    private void drawGrid(Graphics2D g, Rectangle board) {
        g.setStroke(new BasicStroke(1f));
        g.setColor(Theme.alpha(Theme.INK, 16));

        for (int i = 1; i < GRID; i++) {
            float offset = i / (float) GRID;
            int x = Math.round(board.x + offset * board.width);
            int y = Math.round(board.y + offset * board.height);
            g.drawLine(x, board.y + 1, x, board.y + board.height - 1);
            g.drawLine(board.x + 1, y, board.x + board.width - 1, y);
        }

        g.setColor(Theme.alpha(Theme.INK, 34));
        for (int i = 7; i < GRID; i += 7) {
            float offset = i / (float) GRID;
            int x = Math.round(board.x + offset * board.width);
            int y = Math.round(board.y + offset * board.height);
            g.drawLine(x, board.y + 1, x, board.y + board.height - 1);
            g.drawLine(board.x + 1, y, board.x + board.width - 1, y);
        }
    }

    private void drawCornerMarks(Graphics2D g, Rectangle board) {
        int length = 9;
        int inset = 7;
        g.setColor(Theme.alpha(Theme.INK, 90));
        g.setStroke(new BasicStroke(1f));

        int left = board.x + inset;
        int right = board.x + board.width - inset;
        int top = board.y + inset;
        int bottom = board.y + board.height - inset;

        g.drawLine(left, top, left + length, top);
        g.drawLine(left, top, left, top + length);
        g.drawLine(right, top, right - length, top);
        g.drawLine(right, top, right, top + length);
        g.drawLine(left, bottom, left + length, bottom);
        g.drawLine(left, bottom, left, bottom - length);
        g.drawLine(right, bottom, right - length, bottom);
        g.drawLine(right, bottom, right, bottom - length);
    }

    private void drawScreenStroke(Graphics2D g, Rectangle board, Stroke stroke) {
        List<Point2D.Double> points = stroke.points();
        if (points.isEmpty()) {
            return;
        }

        float cell = board.width / (float) GRID;
        g.setColor(Theme.INK);
        g.setStroke(new BasicStroke(
            Math.max(1.2f, stroke.brushCells() * cell),
            BasicStroke.CAP_ROUND,
            BasicStroke.JOIN_ROUND));

        if (points.size() == 1) {
            Point2D.Double point = points.getFirst();
            double radius = stroke.brushCells() * cell / 2.0;
            g.fill(new Ellipse2D.Double(
                board.x + point.x * cell - radius,
                board.y + point.y * cell - radius,
                radius * 2, radius * 2));
            return;
        }

        Path2D.Double path = new Path2D.Double();
        Point2D.Double first = points.getFirst();
        path.moveTo(board.x + first.x * cell, board.y + first.y * cell);

        for (int i = 1; i < points.size(); i++) {
            Point2D.Double point = points.get(i);
            path.lineTo(board.x + point.x * cell, board.y + point.y * cell);
        }

        g.draw(path);
    }

    private void drawRasterStroke(Graphics2D g, Stroke stroke) {
        List<Point2D.Double> points = stroke.points();
        float cell = RASTER / (float) GRID;
        g.setStroke(new BasicStroke(
            stroke.brushCells() * cell,
            BasicStroke.CAP_ROUND,
            BasicStroke.JOIN_ROUND));

        if (points.size() == 1) {
            Point2D.Double point = points.getFirst();
            double radius = stroke.brushCells() * cell / 2.0;
            g.fill(new Ellipse2D.Double(
                point.x * cell - radius, point.y * cell - radius, radius * 2, radius * 2));
            return;
        }

        Path2D.Double path = new Path2D.Double();
        Point2D.Double first = points.getFirst();
        path.moveTo(first.x * cell, first.y * cell);

        for (int i = 1; i < points.size(); i++) {
            Point2D.Double point = points.get(i);
            path.lineTo(point.x * cell, point.y * cell);
        }

        g.draw(path);
    }

    private void drawEmptyState(Graphics2D g, Rectangle board) {
        int centerX = board.x + board.width / 2;
        int centerY = board.y + board.height / 2;

        String title = "Draw a digit";
        g.setFont(emptyTitleFont);
        g.setColor(Theme.alpha(Theme.MUTED, 190));
        g.drawString(title,
            centerX - g.getFontMetrics().stringWidth(title) / 2,
            centerY - 3);

        String hint = "0 — 9";
        g.setFont(emptyHintFont);
        g.setColor(Theme.alpha(Theme.MUTED, 130));
        g.drawString(hint,
            centerX - g.getFontMetrics().stringWidth(hint) / 2,
            centerY + 17);
    }

    private void drawCursor(Graphics2D g, Rectangle board) {
        float cell = board.width / (float) GRID;
        float radius = brushCells * cell / 2f;
        float x = board.x + (float) cursorCell.x * cell;
        float y = board.y + (float) cursorCell.y * cell;

        g.setColor(Theme.alpha(Theme.ACCENT, 26));
        g.fill(new Ellipse2D.Float(x - radius, y - radius, radius * 2, radius * 2));

        g.setStroke(new BasicStroke(1f));
        g.setColor(Theme.alpha(Theme.ACCENT, 200));
        g.draw(new Ellipse2D.Float(x - radius, y - radius, radius * 2, radius * 2));
    }

    private Point2D.Double toCell(Point componentPoint, boolean clamp) {
        Rectangle board = boardBounds();
        if (board.width <= 0) {
            return null;
        }

        if (!clamp && !board.contains(componentPoint)) {
            return null;
        }

        double x = (componentPoint.x - board.x) * GRID / (double) board.width;
        double y = (componentPoint.y - board.y) * GRID / (double) board.height;

        if (clamp) {
            x = Math.max(0, Math.min(GRID, x));
            y = Math.max(0, Math.min(GRID, y));
        } else {
            x = Math.max(0, Math.min(GRID, x));
            y = Math.max(0, Math.min(GRID, y));
        }

        return new Point2D.Double(x, y);
    }

    private Rectangle boardBounds() {
        int size = Math.min(getWidth(), getHeight()) - 8;
        if (size <= 0) {
            return new Rectangle();
        }
        int x = (getWidth() - size) / 2;
        int y = (getHeight() - size) / 2;
        return new Rectangle(x, y, size, size);
    }
}
