import pyqtgraph as pg
import numpy as np
import math

def setup_plot(title, width, height):
    plot_widget = pg.PlotWidget(title=title)
    plot_widget.setBackground('k')  # Dark background
    plot_widget.setLabel('left', title)
    plot_widget.setLabel('bottom', 'Batch')
    plot_widget.addLegend()
    plot_widget.setFixedSize(width, height)
    plot_widget.getAxis('left').setPen('w')
    plot_widget.getAxis('bottom').setPen('w')
    plot_widget.getAxis('left').setTextPen('w')
    plot_widget.getAxis('bottom').setTextPen('w')
    return plot_widget

def update_line_plot(plot_widget, x_data, y_data, name, color):
    if name not in [item.name() for item in plot_widget.listDataItems()]:
        plot_widget.plot(x_data, y_data, name=name, pen=color)
    else:
        for item in plot_widget.listDataItems():
            if item.name() == name:
                item.setData(x_data, y_data)
                break

def update_scatter_plot(plot_widget, x_data, y_data, colors):
    plot_widget.clear()
    scatter = pg.ScatterPlotItem(x=x_data, y=y_data, size=10, pen=None, brush=colors)
    plot_widget.addItem(scatter)




def set_plot_colors(image_view):
    image_view.view.setBackgroundColor('k')
    image_view.ui.histogram.axis.setPen('w')
    image_view.ui.histogram.axis.setTextPen('w')

def set_histogram_preset(image_view):
    image_view.ui.histogram.gradient.loadPreset('viridis')
    image_view.ui.histogram.show()



from pyqtgraph import TextItem

def update_image_plot(image_view, data, title='Image Plot'):
    if image_view is None or data is None:
        return
    
    # Ensure data is 2D
    if data.ndim == 1:
        # Calculate the closest rectangular shape
        side1 = int(math.sqrt(data.size))
        side2 = math.ceil(data.size / side1)
        data_padded = np.zeros(side1 * side2)
        data_padded[:data.size] = data
        data = data_padded.reshape(side1, side2)
    

    data_min, data_max = data.min(), data.max()
    if data_min != data_max:
        data_normalized = (data - data_min) / (data_max - data_min)
    else:
        data_normalized = data - data_min  # If all values are the same, just subtract the minimum
    
    image_view.setImage(data_normalized.T)  
    image_view.setLevels(0, 1)  # Set levels to 0-1 range
    image_view.ui.histogram.gradient.loadPreset('viridis')
    image_view.ui.histogram.show()
    
    # Remove existing title if any
    for item in image_view.view.childItems():
        if isinstance(item, TextItem):
            image_view.view.removeItem(item)
    
    # Add new title
    title_item = TextItem(title, anchor=(0.5, 1), color='w')
    image_view.view.addItem(title_item)
    title_item.setPos(data.shape[1] / 2, 0)
