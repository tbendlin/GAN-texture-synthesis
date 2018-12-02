import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk

"""
    Class file that will create and pull up the GUI for texture selection from a real scene.
    
    :author Theodora Bendlin (trb7281)
"""
class GUI:
    def __init__(self):
        pass

    def start(self):
        root = tk.Tk()
        str_dim = str(root.winfo_screenwidth()) + "x" + str(root.winfo_screenheight())
        root.geometry(str_dim)
        root.title("Image selection window")

        app = Window(root)
        root.mainloop()

class Window(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)

        # Defining main components of GUI
        self.master = master
        self.canvas = None
        self.menu = None
        self.upload = None
        self.select = None

        self.im = None

        # Points for drawing lines
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.x = self.y = 0

        # Initialization function for all widgets
        self.init_window()

    """
        Initialization function for all widgets.
        
        Creates a top "menu" that has available actions, and a canvas on which the user can make a rectangle
        selection from a scene to get a particular texture.
    """
    def init_window(self):

        menu_height = 50
        self.menu = tk.Frame(self.master, bg='white', width=self.master.winfo_screenwidth(), height=menu_height,relief='sunken', background='white')
        self.menu.pack(expand=True, fill='both', side='top')

        self.upload = tk.Button(self.menu, text="Upload", command=self.upload_callback)
        self.upload.pack(side='left')

        self.select = tk.Button(self.menu, text="Select Sample", command=self.select_callback)
        self.select.pack(side='right')

        canvas_height = self.master.winfo_screenheight() - menu_height
        self.canvas = tk.Canvas(self.master, width=self.master.winfo_screenwidth(), height=canvas_height)
        self.canvas.pack(expand=True, fill='both', side='bottom')

    """
        Callback function for when a user clicks on the "Upload" Button
        Brings up a file selection screen where a user can select a .jpg or .png.
        The image will be resized to fit the canvas 
    """
    def upload_callback(self):
        # Opening file dialog for user to select an image
        path = tk.filedialog.askopenfilename(filetypes=[('png images', '.png'), ('jpg images', '.jpg')])

        # Opening the image using PIL
        self.im = Image.open(path)

        # Resizing image to fit canvas dimensions, if necessary
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        image_width, image_height = self.im.size

        if image_width > canvas_width or image_height > canvas_height:
            resize_factor = min(float(canvas_width) / float(image_width), float(canvas_height) / float(image_height))
            new_width, new_height = int(image_width * resize_factor), int(image_height * resize_factor)
            self.im = self.im.resize((new_width, new_height), Image.ANTIALIAS)

        # Creating the image using ImageTk because Tkinter only supports GIFs
        # We need to keep a reference to the image so that it isn't garbage collected and disappear
        tkimage = ImageTk.PhotoImage(self.im)

        # Displaying image in the canvas
        self.canvas.create_image(0, 0, image=tkimage, anchor=tk.NW)
        self.canvas.image = tkimage
        self.canvas.pack()

        # Binding user events to the canvas so we can do the rectangle selection
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    """
        Callback for select button. Saves the user selection to the /output folder in this path and 
        closes the GUI
    """
    def select_callback(self):
        boundaries = self.canvas.bbox(self.rect)
        self.im.crop((boundaries[0], boundaries[1], boundaries[2], boundaries[3])).save("output/sample.png")
        root.destroy()

    """
        Event method that responds to a user clicking on some component of the screen to start
        creating the rectangle. If one already exists, it will be destroyed because the user cannot
        have multiple selections
        
        If the start_x or start_y is outside of the bounds of the image, the rectangle will not be drawn
    """
    def on_button_press(self, event):
        canvasx, canvasy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        im_bound_x, im_bound_y = self.im.size
        if canvasx < im_bound_x and canvasy < im_bound_y:
            # save mouse drag start position
            self.start_x, self.start_y = canvasx, canvasy

            # create rectangle if not yet exist
            if not self.rect:
                self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='black')

    """
        Event method that responds to a user dragging an existing rectangle selection. Selection will stop
        once the user ceases the drag. If the user goes out of bounds of the image, the rectangle will not be drawn
    """
    def on_move_press(self, event):
        currX, currY = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        im_bound_x, im_bound_y = self.im.size
        if currX < im_bound_x and currY < im_bound_y:
            self.canvas.coords(self.rect, self.start_x, self.start_y, currX, currY)

    """
        Button release function. Stop drawing when the user releases
    """
    def on_button_release(self, event):
        pass
