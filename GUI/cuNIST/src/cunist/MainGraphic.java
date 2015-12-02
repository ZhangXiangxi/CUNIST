package cunist;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

/**
 * Created by ZouKaifa on 2015/12/1.
 */
public class MainGraphic extends JFrame {
    private static int n = 10;
    private JPanel rootPanel;
    private WriteCanvas writeCanvas;  //手写
    private DotMatrixCanvas dotMatrixCanvas;  //点阵
    private ResultCanvas resultCanvas;  //结果
    JLabel resultJLabel;  //结果
    JLabel[] numberJLabels;  //十个数字
    JLabel[] proJLabels;  //概率
    private JButton clearButton;  //清除

    public MainGraphic() {
        super("手写数字识别");
        this.initComponents();
        this.setAttr();
        this.addComponents();
        this.addListener();
        this.setContentPane(rootPanel);
        this.setBounds(70, 150, 1200, 388);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setResizable(false);
        this.setVisible(true);
    }

    private void initComponents() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }
        rootPanel = new JPanel();
        dotMatrixCanvas = new DotMatrixCanvas();
        writeCanvas = new WriteCanvas(dotMatrixCanvas, this);
        resultCanvas = new ResultCanvas();
        resultJLabel = new JLabel("?", JLabel.CENTER);
        proJLabels = new JLabel[n];
        numberJLabels = new JLabel[n];
        for (int i = 0; i < n; i++) {
            numberJLabels[i] = new JLabel(String.valueOf(i), JLabel.CENTER);
            proJLabels[i] = new JLabel("0.000000000000", JLabel.LEFT);
        }
        clearButton = new JButton("清空");
    }

    private void setAttr() {
        rootPanel.setLayout(null);
        writeCanvas.setBounds(5, 5, 350, 350);
        dotMatrixCanvas.setBounds(356, 76, 280, 280);
        clearButton.setBounds(370, 10, 120, 55);
        clearButton.setFocusPainted(false);
        clearButton.setContentAreaFilled(false);
        clearButton.setFont(new Font("楷体", Font.BOLD, 25));
        clearButton.setBorder(BorderFactory.createLineBorder(Color.RED));
        resultJLabel.setBounds(545, 5, 70, 70);
        resultJLabel.setFont(new Font("黑体", Font.BOLD, 60));
        resultJLabel.setBorder(BorderFactory.createLineBorder(Color.blue));
        for (int i = 0; i < n; i++) {
            numberJLabels[i].setBounds(640, 5+i*35, 30, 30);
            numberJLabels[i].setFont(new Font("黑体", Font.BOLD, 30));
            proJLabels[i].setBounds(675, 5+i*35, 100, 30);
            proJLabels[i].setFont(new Font("黑体", Font.PLAIN, 14));
        }
    }

    private void addComponents() {
        rootPanel.add(writeCanvas);
        rootPanel.add(dotMatrixCanvas);
        rootPanel.add(clearButton);
        rootPanel.add(resultJLabel);
        for (int i = 0; i < n; i++) {
            rootPanel.add(numberJLabels[i]);
            rootPanel.add(proJLabels[i]);
        }
    }

    private void addListener(){
        clearButton.addMouseListener(new Cursor());
        clearButton.addActionListener((ActionEvent e)->{
            resultJLabel.setText("?");
            writeCanvas.clear();
            for (int i = 0; i < n; i++) {
                numberJLabels[i].setForeground(Color.BLACK);
                proJLabels[i].setText(String.valueOf("0.000000000000"));
            }
        });
    }

    class Cursor extends MouseAdapter{
        @Override
        public void mouseEntered(MouseEvent event){
            MainGraphic.this.setCursor(java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.HAND_CURSOR));
        }

        @Override
        public void mouseExited(MouseEvent e) {
            MainGraphic.this.setCursor(java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.DEFAULT_CURSOR));
        }
    }
}
