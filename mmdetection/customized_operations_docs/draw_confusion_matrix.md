运行命令：

python tools/analysis_tools/confusion_matrix.py \

    .../config.py \

    .../result.pkl \

    .../confusion_matrix/

其中，.../config.py表示配置文件，.../result.pkl表示结果文件，.../confusion_matrix/表示混淆矩阵存放路径，要输出正确格式的混淆矩阵，可能需要修改confusion_matrix.py中plot_confusion_matrix函数定义中的：

color_theme——颜色主题，详见[https://matplotlib.org/stable/tutorials/colors/colormaps.html]()；

fig, ax = plt.subplots(...)——修改图片大小；

title_font——图标题字体；

label_font——图标签字体；

ax.set_xticklabels(labels, size=...)——x轴刻度；

ax.set_yticklabels(labels, size=...)——y轴刻度；

ax.text()——混淆矩阵中数字显示。
