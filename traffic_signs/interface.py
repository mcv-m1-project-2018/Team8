import sys
print(sys.version)
import sys, os
print(os.path.dirname(sys.executable))

from tkinter import Tk, Label, Button, StringVar, Spinbox, \
                    Radiobutton, IntVar, Frame
from tkinter.filedialog import askopenfilename
from tkinter import ttk


from PIL import Image, ImageTk

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Sign Detection")
        master.geometry("1200x800")

        #DIVIDE REGIONS
        self.top = Frame(master, height=600)
        self.bottom = Frame(master, height=200)

        ##TOP SUBSECTIONS
        self.imagesPreview = Frame(self.top, borderwidth=2, relief="solid", width=1000)
        self.statisticalResults = Frame(self.top, borderwidth=2, relief="solid", width=200)

        ##BOTTOM SUBSECTIONS
        self.selectionMenu = Frame(self.bottom, borderwidth=2, relief="solid", width=300)
        self.commands = Frame(self.bottom, borderwidth=2, relief="solid", width=300)

        ###SELECTION MENU
        self.selected = IntVar()
        self.selected.set(1)
        self.selected_photo_path = None
        self.selected_photo_name = None
        self.rad1 = Radiobutton(self.selectionMenu,text='  N-Imatges:', value=1, variable=self.selected)
        self.rad1.grid(column=0, row=0)
        self.spin = Spinbox(self.selectionMenu, from_=-1, to=600, width=10)
        self.spin.grid(column=1,row=0)

        self.rad2 = Radiobutton(self.selectionMenu,text='Escollir-ne:', value=2, variable=self.selected)
        self.rad2.grid(column=0, row=1)
        self.B_select_photo = Button(self.selectionMenu, text="Browse...", width=10)
        self.B_select_photo.configure( command=self.get_image_path)
        self.B_select_photo.grid(column=1, row=1)

        ##COMMAND MENU
        self.B_executar = Button(self.commands, text="RUN", width=10)
        self.B_executar.configure(command=self.executar)
        self.B_executar.grid(column=0, row=0)
        style = ttk.Style()
        style.theme_use('default')
        style.configure("black.Horizontal.TProgressbar", background='black')
        bar = ttk.Progressbar(self.commands, length=100, style='black.Horizontal.TProgressbar')
        bar['value'] = 0
        bar.grid(column=0, row=1)
        self.B_anterior = Button(self.commands, text="anterior", width=10)
        self.B_anterior.configure(command=self.executar)
        self.B_anterior.grid(column=1, row=0)
        self.B_seguent = Button(self.commands, text="seguent", width=10)
        self.B_seguent.configure(command=self.executar)
        self.B_seguent.grid(column=2, row=0)

        ###IMAGES PREVIEW
        tab_control = ttk.Notebook(self.imagesPreview)
        self.tab_original = ttk.Frame(tab_control)
        self.tab_mask = ttk.Frame(tab_control)
        self.tab_result = ttk.Frame(tab_control)
        self.tab_GT = ttk.Frame(tab_control)
        self.tab_diff = ttk.Frame(tab_control)
        tab_control.add(self.tab_original, text='Original')
        tab_control.add(self.tab_mask, text='Mask')
        tab_control.add(self.tab_result, text='Result')
        tab_control.add(self.tab_GT, text='GT', padding=100)
        tab_control.add(self.tab_diff, text='Diff')
        tab_control.pack(expand=1, fill='both', side="right")


        ##FINAL
        self.top.pack(side="top",  fill="both")
        self.bottom.pack(side="bottom", fill="both")
        self.imagesPreview.pack(side="left", fill="both")
        self.statisticalResults.pack(side="right",fill="both")
        self.selectionMenu.pack(side="left")
        self.commands.pack(side="right")

        # self.label = Label(master, textvariable=self.label_text)
        # self.label.bind("<Button-1>", self.cycle_label_text)


        # self.close_button = Button(master, text="Close", command=master.quit)
        # self.close_button.grid(column=0, row=5)

    def clicked(self):
        print(self.selected.get())

    def get_image_path(self):
        filepath = askopenfilename(initialdir = "/",title = "Escollir Imatge",filetypes=(("JPEG images", "*.jpg"),
                                           ("PNG images", "*.png"),
                                           ("GIF images", "*.gif"),
                                           ("All files", "*.*") ))
        pathNname = filepath.split("/")
        _, name = pathNname[:-1], pathNname[-1]
        if(len(name) > 2):
            self.selected_photo_path = filepath
            self.B_select_photo.configure(text=name,  width=10)
            showImage(filepath, self.tab_original)
            self.selected.set(2)
    def set_spin_rad(self):
        self.selected.set(1)
    def executar(self):
        pass
    def anterior(self):
        pass
    def seguent(self):
        pass

def showImage(filepath, window):
    im = Image.open(filepath)
    resized_ratio = im.width/1000
    im = im.resize((int(im.width/resized_ratio),int(im.height/resized_ratio)))
    tkimage = ImageTk.PhotoImage(im)
    img = Label(window, image=tkimage, height=600)
    img.image = tkimage
    img.grid(column=0, row=0)

from subprocess import call
def main():
    root = Tk()
    my_gui = MyFirstGUI(root)
    root.mainloop()

if __name__ == '__main__':
    # read arguments
    main()
