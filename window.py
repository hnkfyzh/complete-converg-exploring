import sys
import numpy as np
import cv2

# Only ask users to install matplotlib if they actually need it
try:
    import matplotlib.pyplot as plt
except:
    print('To display the environment in a window, please install matplotlib, eg:')
    print('pip3 install --user matplotlib')
    sys.exit(-1)

class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig = plt.figure(1,(18.86,13.31
                                 ),tight_layout=True)
        self.ax = plt.subplot()
        # self.fig, self.ax = plt.subplots()
        # self.ax.figure.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.95, hspace=0.6, wspace=0.3)

        # Show the env name in the window title
        self.fig.canvas.manager.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """
        img = img.astype(np.uint8)
        img = cv2.resize(img,(1886,1331),interpolation=cv2.INTER_NEAREST)

        def grayscale_to_color(image, colormap=cv2.COLORMAP_HOT):
            # 将灰度图转换为彩色图
            color_image = cv2.applyColorMap(image, colormap)
            return color_image

        # 转换为彩色图像
        color_img = grayscale_to_color(img)
        cv2.imwrite('map_exploring_process.jpg', color_img)
        cv2.imshow("planimg",color_img)
        cv2.waitKey(1)
        # Show the first image of the environment
        # if self.imshow_obj is None:
        #     self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')
        #
        # self.imshow_obj.set_data(img)
        # self.fig.canvas.draw()
        #
        # # Let matplotlib process UI events
        # # This is needed for interactive mode to work properly
        # plt.pause(0.001)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
