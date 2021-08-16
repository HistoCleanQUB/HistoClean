import sys
import tkinter as tk
import tkinter.font as font
from PIL import ImageTk, Image
import os
import cv2
from skimage import io
from skimage.exposure import match_histograms
import threading
from imgaug import augmenters as iaa
import random
import imageio
import time
from tkinter import filedialog
from tkinter import ttk
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
from patchify import patchify, unpatchify
from numpy import asarray
import multiprocessing
from skimage.filters.rank import core_cy_3d
import imagecorruptions
from numba import jit
#os.add_dll_directory(r'bin')
import openslide


print("Loading...")
# Initiate Splash Screen




# Initiate the application
MainScreen = tk.Tk()

# Get the screen dimensions
ScreenWidth = int(MainScreen.winfo_screenwidth())
ScreenHeight = int(MainScreen.winfo_screenheight())
ScreenWidthMiddle = int(ScreenWidth/2)
ScreenHeightMiddle = int(ScreenHeight/2)

# Calculate number of CPU cores and set this to the number of threads available for multithreading
ThreadsAllowed = int(multiprocessing.cpu_count())

# Set the main screen settings
MainScreen.title("HistoClean") # Window title
MainScreen.geometry(f"{ScreenWidth}x{ScreenHeight}+0+0") # Window size
MainScreen.state('zoomed') # Start maximised
MainScreen.iconbitmap(r"Icon/HistoSquare.ico") # Add icon to top bar

# Font Styles
Large_Font = ("Ariel", 14)
Medium_Font = ("Ariel", 12)
Small_Font = ("Ariel", 10)

# Initialising Global Values
Active_Module = ""
Images_Processed = 0
Tiles_Created = 0
Tiles_Max_Current_Image = 100000

def SaveData():
    global Active_Module
    DefaultFile = [('HistoClean File', '*.hc')]
    SavePath = filedialog.asksaveasfilename(initialdir=os.getcwd(), filetypes = DefaultFile, defaultextension=DefaultFile)
    print(f"Saving data to {SavePath}")
    Data = []

    if Active_Module == "Patch":
        Data.append("Patch")
        #Data.append(SettingValues)

def IntOnly(Char, acttyp):
    if acttyp == "1":
        if not Char.isdigit():
            return False
    return True

def ProgressPopup(MaxValue): # Where Value = Variable for images processed and MaxValue = Length of Image List
    global Images_Processed
    Popup = tk.Toplevel(MainScreen)
    Popup.geometry(f"500x100+{ScreenWidthMiddle - 250}+{ScreenHeightMiddle - 50}")
    Popup.title("Complete")
    Popup.resizable(False, False)
    Popup.iconbitmap(r"Icon/HistoSquare.ico")
    Popup.grab_set()

    Progresslabel = tk.Label(Popup, text="Progress:")
    Progresslabel.place(relx=0.5, rely=0.2, anchor="center")
    ProgressVar = tk.IntVar()
    pgbar = ttk.Progressbar(Popup, length=450, orient="horizontal", maximum=MaxValue, value=0, variable=ProgressVar)
    pgbar.place(anchor="center", relx=0.5, rely=0.5)
    Start = time.time()
    print("Start Time: " + str(Start))
    while Images_Processed < MaxValue:
        #print(f"{Images_Processed}/{MaxValue}")
        pgbar.config(maximum=MaxValue)
        ProgressVar.set(Images_Processed)
        Progresslabel.config(text=f"Progress: {Images_Processed} / {MaxValue} complete")
        Popup.update()
        time.sleep(0.000001)

    Progresslabel.config(text=f"Progress: {Images_Processed} / {MaxValue} complete")
    ProgressVar.set(Images_Processed)
    Popup.update()

    Progresslabel.config(text=f"Finished")
    Popup.update()
    End = time.time()
    print("End Time: " +str(End))
    TimeElapsed = (End-Start)
    print("Time Taken: " + str(TimeElapsed))
    time.sleep(1)
    Images_Processed = 0
    Popup.grab_release()
    Popup.destroy()
    Images_Processed = 0
    print("Done ", Images_Processed )

def DoubleProgressPopup(MaxValueImages):
    global Images_Processed
    global Tiles_Created
    global Tiles_Max_Current_Image
    Popup = tk.Toplevel(MainScreen)
    Popup.geometry(f"500x200+{ScreenWidthMiddle - 250}+{ScreenHeightMiddle - 50}")
    Popup.title("Complete")
    Popup.resizable(False, False)
    Popup.iconbitmap(r"Icon/HistoSquare.ico")
    Popup.grab_set()
    #print(f"Tiles_Max_Current_Image: {Tiles_Max_Current_Image}")

    ProgresslabelImages = tk.Label(Popup, text="Progress - Images:")
    ProgresslabelImages.place(relx=0.5, rely=0.2, anchor="center")
    ProgressVarImages = tk.IntVar()
    pgbarImages = ttk.Progressbar(Popup, length=450, orient="horizontal", maximum=MaxValueImages, value=0, variable=ProgressVarImages)
    pgbarImages.place(anchor="center", relx=0.5, rely=0.4)

    ProgresslabelTiles = tk.Label(Popup, text="Progress - Tiles:")
    ProgresslabelTiles.place(relx=0.5, rely=0.6, anchor="center")
    ProgressVarTiles = tk.IntVar()
    pgbarTiles = ttk.Progressbar(Popup, length=450, orient="horizontal", maximum=10000, value=0, variable=ProgressVarTiles)
    pgbarTiles.place(anchor="center", relx=0.5, rely=0.8)


    Start = time.time()
    #print("Start Time: " + str(Start))

    while Images_Processed < MaxValueImages or Tiles_Created < Tiles_Max_Current_Image:
        #print(f"Images_Processed: {Images_Processed}")
        #print(f"Tiles Created: {Tiles_Created}")
        # print(f"{Images_Processed}/{MaxValue}")
        pgbarTiles.config(maximum=Tiles_Max_Current_Image)
        pgbarImages.config(maximum=MaxValueImages)
        ProgressVarTiles.set(Tiles_Created)
        ProgresslabelTiles.config(text=f"Progress: {Tiles_Created} / {Tiles_Max_Current_Image} Tiles extracted")
        ProgresslabelImages.config(text=f"Progress: {Images_Processed} / {MaxValueImages} Images complete")
        ProgressVarImages.set(Images_Processed)
        Popup.update()

        time.sleep(0.000001)




    ProgresslabelImages.config(text=f"Finished")
    Popup.update()
    End = time.time()
    #print("End Time: " + str(End))
    TimeElapsed = (End - Start)
    #print("Time Taken: " + str(TimeElapsed))
    time.sleep(1)
    Images_Processed = 0
    Tiles_Created = 0
    Popup.grab_release()
    Popup.destroy()
    Images_Processed = 0
    Tiles_Created = 0
    print("Done ", Images_Processed)

def ChangeActiveWindow(ModuleButton):
    global Active_Module
    if ModuleButton == "PatchDown":
        if Active_Module != "Patch":
            for widget in MainFrame.winfo_children():
                widget.destroy()

    if ModuleButton == "WhiteDown":
        if Active_Module != "White":
            for widget in MainFrame.winfo_children():
                widget.destroy()

    if ModuleButton == "BalanceDown":
        if Active_Module != "Balance":
            for widget in MainFrame.winfo_children():
                widget.destroy()

    if ModuleButton == "NormDown":
        if Active_Module != "Norm":
            for widget in MainFrame.winfo_children():
                widget.destroy()

    if ModuleButton == "AugDown":
        if Active_Module != "Aug":
            for widget in MainFrame.winfo_children():
                widget.destroy()



    OptionModuleList = [PatchModuleImageLabel, PatchModuleOptionLabel, PatchModuleOption,
                        WhiteModuleImageLabel, WhiteModuleOptionLabel, WhiteModuleOption,
                        BalanceModuleImageLabel, BalanceModuleOptionLabel, BalanceModuleOption,
                        NormModuleImageLabel, NormModuleOptionLabel, NormModuleOption,
                        AugModuleImageLabel, AugModuleOptionLabel, AugModuleOption]

    for Widget in OptionModuleList:
        Widget.config(bg="gray83")

    if ModuleButton == "PatchDown":
        PatchModuleImageLabel.config(bg="gray60")
        PatchModuleOptionLabel.config(bg="gray60")
        PatchModuleOption.config(bg="gray60", relief="sunken")
    elif ModuleButton == "PatchRelease":
        PatchModuleImageLabel.config(bg="gray60")
        PatchModuleOptionLabel.config(bg="gray60")
        PatchModuleOption.config(bg="gray60", relief="groove")
        print("Patching Selected")
        PatchingModule()

    elif ModuleButton == "WhiteDown":
        WhiteModuleImageLabel.config(bg="gray60")
        WhiteModuleOptionLabel.config(bg="gray60")
        WhiteModuleOption.config(bg="gray60", relief="sunken")
    elif ModuleButton == "WhiteRelease":
        WhiteModuleImageLabel.config(bg="gray60")
        WhiteModuleOptionLabel.config(bg="gray60")
        WhiteModuleOption.config(bg="gray60", relief="groove")
        print("Whitespace Filter Selected")
        WhitespaceModule()

    elif ModuleButton == "BalanceDown":
        BalanceModuleImageLabel.config(bg="gray60")
        BalanceModuleOptionLabel.config(bg="gray60")
        BalanceModuleOption.config(bg="gray60", relief="sunken")
    elif ModuleButton == "BalanceRelease":
        BalanceModuleImageLabel.config(bg="gray60")
        BalanceModuleOptionLabel.config(bg="gray60")
        BalanceModuleOption.config(bg="gray60", relief="groove")
        print("Quick Balance Selected")
        BalanceModule()

    elif ModuleButton == "NormDown":
        NormModuleImageLabel.config(bg="gray60")
        NormModuleOptionLabel.config(bg="gray60")
        NormModuleOption.config(bg="gray60", relief="sunken")
    elif ModuleButton == "NormRelease":
        NormModuleImageLabel.config(bg="gray60")
        NormModuleOptionLabel.config(bg="gray60")
        NormModuleOption.config(bg="gray60", relief="groove")
        print("Normalisation Selected")
        NormalisationModule()

    elif ModuleButton == "AugDown":
        AugModuleImageLabel.config(bg="gray60")
        AugModuleOptionLabel.config(bg="gray60")
        AugModuleOption.config(bg="gray60", relief="sunken")
    elif ModuleButton == "AugRelease":
        AugModuleImageLabel.config(bg="gray60")
        AugModuleOptionLabel.config(bg="gray60")
        AugModuleOption.config(bg="gray60", relief="groove")
        print("Augmentation Selected")
        AugmentationModule()
############### Top Menu Options ###############

# menubar = tk.Menu(MainScreen)
#filemenu = tk.Menu(menubar, tearoff=0)
# filemenu.add_command(label="Load")
# filemenu.add_command(label="Save", command=SaveData)
# filemenu.add_separator()
# filemenu.add_command(label="Exit")
# menubar.add_cascade(label="File (Coming Soon)", menu=filemenu)

#  helpmenu = tk.Menu(menubar, tearoff=0)
#  helpmenu.add_command(label="GitHub")
# helpmenu.add_command(label="Documentation")
#  helpmenu.add_command(label="YouTube")
#  menubar.add_cascade(label="Help (Coming Soon)", menu=helpmenu)

#MainScreen.config(menu=menubar)

##################################################

ModulesFrame = tk.Frame(MainScreen, width=ScreenWidth*0.2, height=ScreenHeight*0.87, borderwidth=2, relief="flat", bg="gray83")
ModulesFrame.place(anchor="w", relx=0.005, rely=0.51)

ModuleOptionLabel = tk.Label(MainScreen, text="Please Select an Option:", font=Large_Font)
ModuleOptionLabel.place(anchor="w", relx=0.05, rely=0.02)

MainFrame = tk.Frame(MainScreen, width=ScreenWidth*0.77, height=ScreenHeight*0.87, borderwidth=2, relief="flat", bg="gray83")
MainFrame.place(anchor="w", relx=0.22, rely=0.51)

### Patch Module Option Bar Button ###
PatchModuleOption = tk.Frame(ModulesFrame, width=(ScreenWidth*0.2)-4, height=ScreenHeight*0.1, bg="gray83", borderwidth=2, relief="groove")
PatchModuleImage = Image.open(r"Icon/Image_Patching_Icon.png")
PatchModuleImage = PatchModuleImage.resize((int(ScreenWidth*0.04), int(ScreenWidth*0.04)), Image.ANTIALIAS)
PatchModuleImage = ImageTk.PhotoImage(PatchModuleImage)
PatchModuleImageLabel = tk.Label(PatchModuleOption, image=PatchModuleImage, width=ScreenWidth*0.04, height=ScreenWidth*0.04, bg="gray83")
PatchModuleImageLabel.image = PatchModuleImage
PatchModuleImageLabel.place(relx=0.15, rely=0.5, anchor="center")
PatchModuleOption.place(anchor="nw", relx=0.0, rely=0.0)
PatchModuleOptionLabel = tk.Label(PatchModuleOption, text="Image Patching", fg="black", bg="gray83", font=Large_Font)
PatchModuleOptionLabel.place(relx=0.6, rely=0.5, anchor="center")
PatchModuleOption.bind("<Button-1>", lambda e: ChangeActiveWindow("PatchDown"))
PatchModuleOption.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("PatchRelease"))
PatchModuleImageLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("PatchDown"))
PatchModuleImageLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("PatchRelease"))
PatchModuleOptionLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("PatchDown"))
PatchModuleOptionLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("PatchRelease"))


### Whitespace Module Option Bar Button ###
WhiteModuleOption = tk.Frame(ModulesFrame, width=(ScreenWidth*0.2)-4, height=(ScreenHeight*0.1)-2, bg="gray83", borderwidth=2, relief="groove")
WhiteModuleImage = Image.open(r"Icon/Whitespace_Filter_Icon.png")
WhiteModuleImage = WhiteModuleImage.resize((int(ScreenWidth*0.04), int(ScreenWidth*0.04)), Image.ANTIALIAS)
WhiteModuleImage = ImageTk.PhotoImage(WhiteModuleImage)
WhiteModuleImageLabel = tk.Label(WhiteModuleOption, image=WhiteModuleImage, width=ScreenWidth*0.04, height=ScreenWidth*0.04, bg="gray83")
WhiteModuleImageLabel.image = WhiteModuleImage
WhiteModuleImageLabel.place(relx=0.15, rely=0.5, anchor="center")
WhiteModuleOption.place(anchor="nw", relx=0.0, rely=0.12)
WhiteModuleOptionLabel = tk.Label(WhiteModuleOption, text="Whitespace Filter", fg="black", bg="gray83", font=Large_Font)
WhiteModuleOptionLabel.place(relx=0.6, rely=0.5, anchor="center")
WhiteModuleOption.bind("<Button-1>", lambda e: ChangeActiveWindow("WhiteDown"))
WhiteModuleOption.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("WhiteRelease"))
WhiteModuleImageLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("WhiteDown"))
WhiteModuleImageLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("WhiteRelease"))
WhiteModuleOptionLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("WhiteDown"))
WhiteModuleOptionLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("WhiteRelease"))

### Balance Module Option Bar Button ###
BalanceModuleOption = tk.Frame(ModulesFrame, width=(ScreenWidth*0.2)-4, height=(ScreenHeight*0.1)-2, bg="gray83", borderwidth=2, relief="groove")
BalanceModuleImage = Image.open(r"Icon/Balanced_Class_Icon.png")
BalanceModuleImage = BalanceModuleImage.resize((int(ScreenWidth*0.05), int(ScreenWidth*0.05)), Image.ANTIALIAS)
BalanceModuleImage = ImageTk.PhotoImage(BalanceModuleImage)
BalanceModuleImageLabel = tk.Label(BalanceModuleOption, image=BalanceModuleImage, width=ScreenWidth*0.05, height=ScreenWidth*0.05, bg="gray83")
BalanceModuleImageLabel.image = BalanceModuleImage
BalanceModuleImageLabel.place(relx=0.15, rely=0.5, anchor="center")
BalanceModuleOption.place(anchor="nw", relx=0.0, rely=0.24)
BalanceModuleOptionLabel = tk.Label(BalanceModuleOption, text="Quick Class Balance", fg="black", bg="gray83", font=Large_Font)
BalanceModuleOptionLabel.place(relx=0.6, rely=0.5, anchor="center")
BalanceModuleOption.bind("<Button-1>", lambda e: ChangeActiveWindow("BalanceDown"))
BalanceModuleOption.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("BalanceRelease"))
BalanceModuleImageLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("BalanceDown"))
BalanceModuleImageLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("BalanceRelease"))
BalanceModuleOptionLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("BalanceDown"))
BalanceModuleOptionLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("BalanceRelease"))

### Normalisation Module Option Bar Button ###
NormModuleOption = tk.Frame(ModulesFrame, width=(ScreenWidth*0.2)-4, height=(ScreenHeight*0.1)-2, bg="gray83", borderwidth=2, relief="groove")
NormModuleImage = Image.open(r"Icon/Norm_Class_Icon.png")
NormModuleImage = NormModuleImage.resize((int(ScreenWidth*0.05), int(ScreenWidth*0.05)), Image.ANTIALIAS)
NormModuleImage = ImageTk.PhotoImage(NormModuleImage)
NormModuleImageLabel = tk.Label(NormModuleOption, image=NormModuleImage, width=ScreenWidth*0.05, height=ScreenWidth*0.05, bg="gray83")
NormModuleImageLabel.image = NormModuleImage
NormModuleImageLabel.place(relx=0.15, rely=0.5, anchor="center")
NormModuleOption.place(anchor="nw", relx=0.0, rely=0.36)
NormModuleOptionLabel = tk.Label(NormModuleOption, text="Image Normalisation", fg="black", bg="gray83", font=Large_Font)
NormModuleOptionLabel.place(relx=0.6, rely=0.5, anchor="center")
NormModuleOption.bind("<Button-1>", lambda e: ChangeActiveWindow("NormDown"))
NormModuleOption.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("NormRelease"))
NormModuleImageLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("NormDown"))
NormModuleImageLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("NormRelease"))
NormModuleOptionLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("NormDown"))
NormModuleOptionLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("NormRelease"))

### Augmentation Module Option Bar Button ###
AugModuleOption = tk.Frame(ModulesFrame, width=(ScreenWidth*0.2)-4, height=(ScreenHeight*0.1)-2, bg="gray83", borderwidth=2, relief="groove")
AugModuleImage = Image.open(r"Icon/Augmentation_Icon.png")
AugModuleImage = AugModuleImage.resize((int(ScreenWidth*0.05), int(ScreenWidth*0.05)), Image.ANTIALIAS)
AugModuleImage = ImageTk.PhotoImage(AugModuleImage)
AugModuleImageLabel = tk.Label(AugModuleOption, image=AugModuleImage, width=ScreenWidth*0.05, height=ScreenWidth*0.05, bg="gray83")
AugModuleImageLabel.image = AugModuleImage
AugModuleImageLabel.place(relx=0.15, rely=0.5, anchor="center")
AugModuleOption.place(anchor="nw", relx=0.0, rely=0.48)
AugModuleOptionLabel = tk.Label(AugModuleOption, text="Image Augmentation\n and Preprocessing", fg="black", bg="gray83", font=Large_Font)
AugModuleOptionLabel.place(relx=0.6, rely=0.5, anchor="center")
AugModuleOption.bind("<Button-1>", lambda e: ChangeActiveWindow("AugDown"))
AugModuleOption.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("AugRelease"))
AugModuleImageLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("AugDown"))
AugModuleImageLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("AugRelease"))
AugModuleOptionLabel.bind("<Button-1>", lambda e: ChangeActiveWindow("AugDown"))
AugModuleOptionLabel.bind("<ButtonRelease-1>", lambda e: ChangeActiveWindow("AugRelease"))





def PatchingModule():
    global Active_Module
    Active_Module = "Patch"
    # Initialise Module Memory
    InputImages = []
    OriginalWidthList = []
    OriginalHeightList = []
    ImageLocationXList = []
    ImageLocationYList = []
    DownfactorList = []
    ThumbnailImages = []
    ThumbnailWidthList = []
    ThumbnailHeightList = []
    DrawnLines = []
    SmallList = []

    CurrentImage = 0
    CanvasWidth = 650
    CanvasHeight = 300
    Images_Loaded = 0
    CurrentImage = 0

    ListList = [InputImages, OriginalWidthList, OriginalHeightList, ImageLocationXList, ImageLocationYList,
                DownfactorList,
                ThumbnailImages, ThumbnailWidthList, ThumbnailHeightList, DrawnLines]

    def getImageThumbnail(ImagePath):
        Slide = openslide.OpenSlide(ImagePath)
        Thumb = Slide.get_thumbnail((CanvasWidth, CanvasHeight))
        print(Thumb)

        Thumb = ImageTk.PhotoImage(Thumb)

        return Thumb


    def EnterLoadImages():
        Step1Text.config(fg="black", text="Select your image folder:")
        nonlocal CurrentImage
        # Reset all inputs


        for li in ListList:
            li.clear()

        filename2 = Step1Entry.get()
        if os.path.isdir(filename2):

            for r, d, f in os.walk(Step1Entry.get()):
                for file in f:
                    extension = file.split(".")
                    extension = extension[-1]
                    print(extension)
                    if 'svs' in extension:
                        InputImages.append(os.path.join(r, file))

            ImageDetectionLabel.config(text=f"{len(InputImages)} Images Detected")
            First_Image = getImageThumbnail(InputImages[0])
            print(First_Image)
            PreviewCanvas.image = First_Image
            PreviewCanvas.config(image=First_Image)

        else:
            Step1Text.config(fg="red", text="Please input a valid folder")
    # Get the images and make into thumbnails
    def LoadImages():
        Step1Text.config(fg="black", text="Select your image folder:")
        nonlocal CurrentImage
        # Reset all inputs
        Step1Entry.delete(0, 10000)
        Step1Text.config(fg="black", text="Select your image folder:")
        for li in ListList:
            li.clear()
        filename2 = filedialog.askdirectory(initialdir=os.getcwd(), title="Please select your top-level folder.")
        Step1Entry.insert(0, filename2)
        if os.path.isdir(filename2):

            for r, d, f in os.walk(Step1Entry.get()):
                for file in f:
                    extension = file.split(".")
                    extension = extension[-1]
                    print(extension)
                    if 'svs' in extension or 'zvi' in extension or 'ndpi' in extension:
                        print(os.path.join(r, file))
                        InputImages.append(os.path.join(r, file))
            print(InputImages)

            ImageDetectionLabel.config(text=f"{len(InputImages)} Images Detected")
            First_Image = getImageThumbnail(InputImages[0])
            print(First_Image)
            PreviewCanvas.image = First_Image
            PreviewCanvas.config(image=First_Image)

            print("Images Loaded Successfully")



        else:
            Step1Text.config(fg="red", text="Please Select a Valid Folder")

        # Draw the lines showing where the image is split up

    # Move to next Image
    def NextImage():
        nonlocal CurrentImage
        CurrentImage += 1
        print(CurrentImage)
        try:
            Im = getImageThumbnail(InputImages[CurrentImage])
            PreviewCanvas.image = Im
            PreviewCanvas.config(image=Im)
        except:
            e = sys.exc_info()
            print(e)
            CurrentImage -= 1

    # Move to previous Image
    def PrevImage():
        nonlocal CurrentImage
        CurrentImage -= 1
        print(CurrentImage)
        if CurrentImage >=0:
            Im = getImageThumbnail(InputImages[CurrentImage])
            PreviewCanvas.image = Im
            PreviewCanvas.config(image=Im)
        else:
            CurrentImage += 1

    # Get the save folder location
    def GetSaveFolder():
        SaveLabel.config(fg="black", text="Select save location")
        files = []
        SaveEntry.delete(0, 1000)
        filename2 = filedialog.askdirectory(initialdir=os.getcwd(), title="Please select your top-level folder.")
        SaveEntry.insert(0, filename2)

        for r, d, f in os.walk(SaveEntry.get()):
            for direct in d:
                files.append(direct)
            for file in f:
                files.append(file)
        if len(files) > 0:
            WarningLabel.place(anchor="center", relx=0.5, rely=0.86)
        else:
            try:
                WarningLabel.place_forget()
            except:
                pass

    # Chek all entries are filled
    def PreChecks():
        Errors = 0
        Step1Text.config(fg="black")
        SizeLabel.config(fg="black")
        SaveLabel.config(fg="black")

        if not os.path.isdir(Step1Entry.get()):
            Step1Text.config(fg="red", text="Please Enter a valid directory")
            Errors += 1
        if len(Step1Entry.get()) < 1:
            Step1Text.config(fg="red")
            Errors += 1
        if len(WidthEntry.get()) < 1:
            SizeLabel.config(fg="red")
            Errors += 1
        if len(SaveEntry.get()) < 1:
            SaveLabel.config(fg="red")
            Errors += 1
        if not all(i >= int(WidthEntry.get()) for i in OriginalWidthList) or not all(i >= int(WidthEntry.get()) for i in OriginalHeightList):
            ImageSizeLabel.config(fg="red", text=f"Warning: Patch size exceeds image size for some images")
            Errors += 1

        if not os.path.isdir(SaveEntry.get()):
            SaveLabel.config(fg="red", text="Please enter a valid directory")
            Errors += 1

        if Errors == 0:
            print("No Errors, Startine Patching")
            PatchThreadThreading()

    # Make folder for each image
    def RecreateFolderStructure():
        print("Recreating the Folder Structure")
        OriginalTopFolder = Step1Entry.get()
        NewSetTopFolder = SaveEntry.get()

        for In in InputImages:
            OriginalImageName = (os.path.split(In)[1]).split(".")[0]
            OriginalImageLocation = os.path.split(In)[0]
            OriginalImageShortLocation = OriginalImageLocation.replace(OriginalTopFolder, "")
            # os.path.join kept fucking up here, no idea why.
            NewFolder = str(NewSetTopFolder) + str(OriginalImageShortLocation) + "\\" + str(OriginalImageName)

            try:
                os.makedirs(NewFolder)
            except:
                pass
        print("Foldrs Made")

    def Update_Progress_Img():
        nonlocal Images_Loaded
        nonlocal InputImages
        while Images_Loaded < len(InputImages):
            print(f"{Images_Loaded}/{len(InputImages)}")
            pgbarImg.config(maximum=len(InputImages))
            StartButton.config(state="disabled")
            ProgressVarImg.set(Images_Loaded)
            # Progresslabel.config(text=f"Progress: {Images_Loaded} / {len(InputImages)} complete")
            MainFrame.update()
            time.sleep(0.000001)
        pgbarImg.place_forget()
        StartButton.config(state="normal")
        PrevButton.place(anchor="center", relx=0.25, rely=0.35)
        NextButton.place(anchor="center", relx=0.75, rely=0.35)

    def PatchThreadThreading():
        print("Spicy Thread Running")
        RecreateFolderStructure()
        t = threading.Thread(target=PatchCore)
        t.start()

        DoubleProgressPopup(len(InputImages))



    def RegionRequest(Slide, XCenter, YCenter, Size,  Tilename, OutPath):
        global Tiles_Created
        #print("In Region Request")
        #print(Slide, XCenter, YCenter, Size)
        #print(Tilename, OutPath)

        Tile = Slide.read_region((int(XCenter), int(YCenter)), level=0, size=(int(Size), int(Size))).convert(
            'RGB').resize((int(WidthEntry.get()),int(WidthEntry.get())))
        Tile.save(OutPath)
        print("Tile Saved SUccesssfuly")

        Tiles_Created += 1

    #Ima is a filepath
    def PatchCore():

        global Images_Processed
        global Tiles_Max_Current_Image
        global Tiles_Created
        NewSetTopFolder = SaveEntry.get()
        OriginalTopFolder = Step1Entry.get()

        for Ima in InputImages:
            print(f"Processing {Ima}")
            ImageName = os.path.split(Ima)[-1]

            OriginalImageLocation = os.path.split(Ima)[0]
            OriginalLocation_NoTopFolder = OriginalImageLocation.replace(OriginalTopFolder, "")
            ImageName_NoEXT = ImageName.split(".")[0]
            Image_Extension = ImageName.split(".")[-1]


            Im = openslide.OpenSlide(Ima)


            Image_Width = Im.dimensions[0]
            Image_Height = Im.dimensions[1]
            #print(Image_Width, Image_Height)

            if Image_Width < int(WidthEntry.get()) or Image_Height< int(WidthEntry.get()):
                SmallList.append(ImageName)
                Images_Processed += 1
            else:
                print("We are go")

                Metadata = Im.properties
                print(Metadata)


                ## Calculate Needed TileSize
                try:
                    Metadata = Im.properties
                    if Image_Extension == "svs":
                        Magnification = int(Metadata["aperio.AppMag"])
                    elif Image_Extension == "ndpi":
                        Magnification = int(Metadata["hamamatsu.SourceLens"])



                except:
                    Magnification = 40

                print(Magnification)
                RealPixels = (Magnification/int(MagEntry.get())) * int(WidthEntry.get())
                print(f"RealPixels: {RealPixels}")



                if ExtentionVar.get() == 0:
                    Extension = ".jpg"

                elif ExtentionVar.get() == 1:
                    Extension = ".png"

                elif ExtentionVar.get() == 2:
                    Extension = ".tif"

                print(f"Extension:{Extension}")

                Centers_x = []
                Centers_y = []
                First_X = 0
                First_Y = 0

                Number_Of_Tiles_X = math.floor(Image_Width / RealPixels)
                Number_Of_Tiles_Y = math.floor(Image_Height / RealPixels)

                Centers_x.append(First_X)
                Centers_y.append(First_Y)

                Current_X = First_X
                Current_Y = First_Y

                for x in range(Number_Of_Tiles_X):
                    XCO = Current_X + (x * RealPixels)

                    for y in range(Number_Of_Tiles_Y):
                        YCO = Current_Y + (y * RealPixels)
                        Centers_x.append(XCO)
                        Centers_y.append(YCO)

                Tiles_Max_Current_Image = len(Centers_x)
                pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
                for x in range(len(Centers_x)):

                    Foldername = str(NewSetTopFolder) + str(OriginalLocation_NoTopFolder)
                    TileName = f"{ImageName_NoEXT}, x={int(Centers_x[x])}, y={int(Centers_y[x])}, Mag={MagEntry.get()}" + Extension
                    Outfilename = os.path.join(Foldername, ImageName_NoEXT, TileName)
                    pool.submit(RegionRequest, Im, Centers_x[x],Centers_y[x], RealPixels,  TileName, Outfilename)
                pool.shutdown(wait=True)

                Images_Processed += 1
                Tiles_Created = 0
        Images_Processed = len(InputImages)
        Tiles_Created = 999999999999999


    InfoLabel = tk.Label(MainFrame, text=f"This module will allow you to tile up your images", font=Large_Font, bg="gray83")
    InfoLabel.place(anchor="center", relx=0.5, rely=0.02)

    Step1Text = tk.Label(MainFrame, fg="black", text="Select your image folder:", font=Medium_Font, bg="gray83")
    Step1Text.place(anchor="center", relx=0.5, rely=0.07)

    Step1Entry = tk.Entry(MainFrame, width=30, font=Large_Font)
    Step1Entry.place(anchor="center", relx=0.475, rely=0.1)
    Step1Entry.bind('<Return>', lambda e: EnterLoadImages())

    ButtonForFiles = tk.Button(MainFrame, text="Browse...", font=Small_Font, command=LoadImages)
    ButtonForFiles.place(anchor="center", relx=0.615, rely=0.1)

    ImageDetectionLabel = tk.Label(MainFrame, fg="blue", text="", bg="gray83", font=Medium_Font)
    ImageDetectionLabel.place(anchor="center", relx=0.5, rely=0.13)

    PreviewCanvas = tk.Label(MainFrame, height=int(ScreenHeight*0.3), width=int(ScreenWidth*0.35), bg="gray83", bd=0, highlightthickness=0)
    PreviewCanvas.place(anchor="n", relx=0.5, rely=0.15)

    ImageSizeLabel = tk.Label(MainFrame, text="", fg="blue", font=Medium_Font, bg="gray83")
    ImageSizeLabel.place(anchor="center", relx=0.5, rely=0.53)

    PrevButton = tk.Button(MainFrame, text="Previous", font=Large_Font, width=10, command=PrevImage)
    PrevButton.place(anchor="center", relx=0.2, rely=0.35)

    NextButton = tk.Button(MainFrame, text="Next", font=Large_Font, width=10, command=NextImage)
    NextButton.place(anchor="center", relx=0.8, rely=0.35)

    SizeLabel = tk.Label(MainFrame, text="Output tile size (pixels)", bg="gray83", font=Medium_Font)
    SizeLabel.place(anchor="center", relx=0.4, rely=0.56)

    Validation = (MainFrame.register(IntOnly))
    WidthEntry = tk.Entry(MainFrame, width=5, justify="center", validate="key", validatecommand=(Validation, "%P", "%d"), font=Large_Font)
    WidthEntry.place(anchor="center", relx=0.4, rely=0.59)

    MagLabel = tk.Label(MainFrame, text="Output Magnification", bg="gray83", font=Medium_Font)
    MagLabel.place(anchor="center", relx=0.6, rely=0.56)
    MagEntry = tk.Entry(MainFrame, width=5, justify="center", validate="key",
                          validatecommand=(Validation, "%P", "%d"), font=Large_Font)
    MagEntry.place(anchor="center", relx=0.6, rely=0.59)

    ExtentionLabel = tk.Label(MainFrame, text="Select your image output extension", bg="gray83", font=Medium_Font)
    ExtentionLabel.place(relx=0.5, rely=0.63, anchor="center")

    ExtentionVar = tk.IntVar()
    JPGExtention = tk.Radiobutton(MainFrame, text=".JPG", var=ExtentionVar, value=0, font=Medium_Font, bg="gray83", activebackground="gray83")
    JPGExtention.place(relx=0.4, rely=0.67, anchor="center")
    PNGExtention = tk.Radiobutton(MainFrame, text=".PNG", var=ExtentionVar, value=1, font=Medium_Font, bg="gray83", activebackground="gray83")
    PNGExtention.place(relx=0.5, rely=0.67, anchor="center")
    TIFExtention = tk.Radiobutton(MainFrame, text=".TIF", var=ExtentionVar, value=2, font=Medium_Font, bg="gray83",
                                  activebackground="gray83")
    TIFExtention.place(relx=0.6, rely=0.67, anchor="center")


    #PreviewButton = tk.Button(MainFrame, text="Preview", font=Medium_Font, command=PreviewGrid)
    #reviewButton.place(anchor="center", relx=0.5, rely=0.72)

    SaveLabel = tk.Label(MainFrame, text="Select save location", width=40, font=Medium_Font, bg="gray83")
    SaveLabel.place(anchor="center", relx=0.5, rely=0.76)

    SaveEntry = tk.Entry(MainFrame,  bg="white", width=30, font=Large_Font)
    SaveEntry.place(anchor="center", relx=0.475, rely=0.8)

    SaveBrowse = tk.Button(MainFrame, text="Browse...", font=Small_Font, command=GetSaveFolder)
    SaveBrowse.place(anchor="center", relx=0.615, rely=0.8)

    WarningLabel = tk.Label(MainFrame,
                            text="WARNING - Save directory is not empty. Images may not save correctly.", fg="red")

    StartButton = tk.Button(MainFrame, text="Start", width=8, font=Small_Font, command=PreChecks)
    StartButton.place(anchor="center", relx=0.5, rely=0.9)


    ProgressVarImg = tk.IntVar()
    pgbarImg = ttk.Progressbar(MainFrame, length=600, orient="horizontal", maximum=100, value=0,
                               variable=ProgressVarImg)

def WhitespaceModule():
    global Active_Module
    Active_Module = "White"
    ImagePaths = []
    ShownImagesPaths = []
    ShownImagesThumbnails = []
    ShownImagesThumbnailsSizes = []
    ShownImageThumbnailMasks = []
    ShownImagePositivePercentage = []
    ShownImageLocation = 4

    def Change_Slider(event=None):  # Changes the slider according to what is entered in the %Tissue Cutoff
        try:
            if int(Step2Entry.get()) > 100:
                Step2Entry.delete(0, 10)
                Step2Entry.insert(0, "100")
            Step2Scale.set(Step2Entry.get())
        except:
            pass

    def Change_Tissue_Entry(val):  # Changes the tissue % box according to slider
        Step2Entry.delete(0, 100)
        Step2Entry.insert(0, val)

    def BringUpOptions():

        Binary_Option_Label.config(fg="black")
        BlurLabel.config(fg="black")
        Block_Size_Label.config(fg="black")
        Anchor_Label.config(fg="black")

        print(TypeDropdown.get())
        Binary_Option_Entry.place_forget()
        Binary_Option_Label.place_forget()
        BlurLabel.place_forget()
        BlurEntry.place_forget()
        Block_Size_Label.place_forget()
        Block_Size_Entry.place_forget()
        Anchor_Label.place_forget()
        Anchor_Entry.place_forget()

        if TypeDropdown.get() == "Positive Pixel Classification":
            Binary_Option_Label.place(relx=0.14, rely=0.4, anchor="center")
            Binary_Option_Entry.place(relx=0.2, rely=0.4, anchor="center")
            BlurLabel.place(relx=0.12, rely=0.45, anchor="center")
            BlurEntry.place(relx=0.2, rely=0.45, anchor="center")
            Preview_Button.place(relx=0.17, rely=0.5)
            HelpLabel.place(relx=0.17, rely=0.55)
            Info_Label.place_configure(relx=0.17, rely=0.6)
            Info_Label.config(text="Pixel classed as positive or negative based on a single threshold.\n"
                                   "The higher the threshold, the lighter the pixels included.")

        elif TypeDropdown.get() == "Adaptive Threshold - Mean":
            Block_Size_Label.place(relx=0.14, rely=0.4, anchor="center")
            Block_Size_Entry.place(relx=0.2, rely=0.4, anchor="center")
            Anchor_Label.place(relx=0.14, rely=0.45, anchor="center")
            Anchor_Entry.place(relx=0.2, rely=0.45, anchor="center")
            BlurLabel.place(relx=0.12, rely=0.5, anchor="center")
            BlurEntry.place(relx=0.2, rely=0.5, anchor="center")
            Preview_Button.place(relx=0.17, rely=0.55)
            HelpLabel.place(relx=0.17, rely=0.6)
            Info_Label.place(relx=0.17, rely=0.65)
            Info_Label.config(
                text="Uses an algorithm to calculate thresholds for small regions of the image (Block size).\n"
                     "This creates different thresholds for different regions of the image.\n"
                     "The threshold value is the mean of each block")

        elif TypeDropdown.get() == "Adaptive Threshold - Gaussian":
            Block_Size_Label.place(relx=0.14, rely=0.4, anchor="center")
            Block_Size_Entry.place(relx=0.2, rely=0.4, anchor="center")
            Anchor_Label.place(relx=0.14, rely=0.45, anchor="center")
            Anchor_Entry.place(relx=0.2, rely=0.45, anchor="center")
            BlurLabel.place(relx=0.12, rely=0.5, anchor="center")
            BlurEntry.place(relx=0.2, rely=0.5, anchor="center")
            Preview_Button.place(relx=0.17, rely=0.55)
            HelpLabel.place(relx=0.17, rely=0.6)
            Info_Label.place(relx=0.17, rely=0.65)
            Info_Label.config(
                text="Uses an algorithm to calculate thresholds for small regions of the image (Block size).\n"
                     "This creates different thresholds for different regions of the image.\n"
                     "The threshold value is the weighted sum of Block values\n"
                     " where weights are a gaussian window.")

        elif TypeDropdown.get() == "Otsu Binarisation":
            BlurLabel.place(relx=0.12, rely=0.4, anchor="center")
            BlurEntry.place(relx=0.2, rely=0.4, anchor="center")
            Preview_Button.place(relx=0.17, rely=0.45)
            HelpLabel.place(relx=0.17, rely=0.5)
            Info_Label.place(relx=0.17, rely=0.55)
            Info_Label.config(text="Creates a histogram for each image and exhaustively searches\n"
                                   "for the minimum between the signal and background.")

    def LoadImages():
        nonlocal ShownImageLocation
        Step1Entry.delete(0, 10000)
        ImagePaths.clear()
        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        ShownImageThumbnailMasks.clear()
        ShownImagePositivePercentage.clear()
        filename2 = filedialog.askdirectory(initialdir=os.getcwd(), title="Please select your top-level folder.")
        Step1Entry.insert(0, filename2)
        for r, d, f in os.walk(Step1Entry.get()):
            for file in f:
                if '.jpg' in file or '.jfif' in file or ".png" or ".tif"  in file:
                    ImagePaths.append(os.path.join(r, file))

        ActiveImages = ImagePaths[:ShownImageLocation]
        for A in ActiveImages:
            ShownImagesPaths.append(A)

        # Make Thumbnail Images
        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            print(type(Im))
            CanvasSize = 230, 230
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            print(ThumbnailSize)

            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)

        CanvasXLoc = 0

        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 1000 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 3)
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            print(width, height)
            OrigCanvas.create_image(CanvasXLoc, 130, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

    def PreChecks():
        Errors = 0
        Binary_Option_Label.config(fg="black")
        BlurLabel.config(fg="black")
        Block_Size_Label.config(fg="black")
        Anchor_Label.config(fg="black")
        Step1Text.config(fg="black", text="Step 1: Select your working directory. (Root folder containing images)")
        Step2Text.config(fg="black")

        if len(Step1Entry.get()) == 0:
            Step1Text.config(fg="red")
            Errors += 1
        elif not os.path.isdir(Step1Entry.get()):
            Step1Text.config(text="Please select a valid folder", fg="red")
            Errors += 1

        if len(Step2Entry.get()) == 0:
            Step2Text.config(fg="red")
            Errors += 1

        if TypeDropdown.get() == "Positive Pixel Classification":
            if len(Binary_Option_Entry.get()) == 0:
                Binary_Option_Label.config(fg="red")
                Errors += 1

            if len(BlurEntry.get()) == 0:
                BlurLabel.config(fg="red")
                Errors += 1

            try:
                float(Binary_Option_Entry.get())
            except:
                Binary_Option_Label.config(fg="red")
                Errors += 1

            if float(Binary_Option_Entry.get()) > 1 or float(Binary_Option_Entry.get()) < 0:
                Binary_Option_Label.config(fg="red")
                Errors += 1
            print(len(BlurEntry.get()))
            if len(BlurEntry.get()) == 0:
                BlurLabel.config(fg="red")
                Errors += 1

            try:
                int(BlurEntry.get())
            except:
                BlurLabel.config(fg="red")
                Errors += 1

            if int(BlurEntry.get()) % 2 == 0:
                if int(BlurEntry.get()) !=0:
                    BlurLabel.config(fg="red")
                    Errors += 1
                    print("here")



        elif TypeDropdown.get() == "Adaptive Threshold - Mean":

            if len(BlurEntry.get()) == 0:
                BlurLabel.config(fg="red")
                Errors += 1

            elif int(BlurEntry.get()) % 2 == 0:
                if int(BlurEntry.get()) != 0:
                    BlurLabel.config(fg="red")
                    Errors += 1

            try:
                int(BlurEntry.get())
            except:
                BlurLabel.config(fg="red")
                Errors += 1

            if len(Anchor_Entry.get()) == 0:
                Anchor_Label.config(fg="red")
                Errors += 1
            elif int(Anchor_Entry.get()) < -100 or int(Anchor_Entry.get()) > 100:
                Errors += 1
                Anchor_Label.config(fg="red")

            if len(Block_Size_Entry.get()) == 0:
                Errors += 1
                Block_Size_Label.config(fg="red")


            elif int(Block_Size_Entry.get()) < 1:
                Errors += 1
                Block_Size_Label.config(fg="red")

            elif int(Block_Size_Entry.get()) % 2 == 0:
                Errors += 1
                Block_Size_Label.config(fg="red")



        elif TypeDropdown.get() == "Adaptive Threshold - Gaussian":
            if len(BlurEntry.get()) == 0:
                BlurLabel.config(fg="red")
                Errors += 1


            elif int(BlurEntry.get()) % 2 == 0:

                if int(BlurEntry.get()) != 0:
                    BlurLabel.config(fg="red")

                    Errors += 1

            try:

                int(BlurEntry.get())

            except:

                BlurLabel.config(fg="red")

                Errors += 1

            if len(Anchor_Entry.get()) == 0:
                Anchor_Label.config(fg="red")
                Errors += 1
            elif int(Anchor_Entry.get()) < -100 or int(Anchor_Entry.get()) > 100:
                Errors += 1
                Anchor_Label.config(fg="red")

            if len(Block_Size_Entry.get()) == 0:
                Errors += 1
                Block_Size_Label.config(fg="red")


            elif int(Block_Size_Entry.get()) < 1:
                Errors += 1
                Block_Size_Label.config(fg="red")

            elif int(Block_Size_Entry.get()) % 2 == 0:
                Errors += 1
                Block_Size_Label.config(fg="red")

        elif TypeDropdown.get() == "Otsu Binarisation":

            if len(BlurEntry.get()) == 0:
                BlurLabel.config(fg="red")
                Errors += 1
            elif int(BlurEntry.get()) % 2 == 0:
                BlurLabel.config(fg="red")
                Errors += 1
        return Errors

    def CreateMask():
        Errors = PreChecks()
        if Errors == 0:
            ShownImageThumbnailMasks.clear()
            ShownImagePositivePercentage.clear()

            try:
                MaskCanvas.delete("all")
            except:
                pass

            if TypeDropdown.get() == "Positive Pixel Classification":
                Cutoff = float(Binary_Option_Entry.get())

                Blur = int(BlurEntry.get())

                print(Blur)

                for I in ShownImagesPaths:
                    img = cv2.imread(I, 0)
                    aug = iaa.AverageBlur(k=(Blur, Blur))
                    img = aug.augment_image(img)

                    ret, outimg = cv2.threshold(img, round(Cutoff * 255), 255, cv2.THRESH_BINARY)
                    width, height = outimg.shape
                    ImageArea = (width * height) * 255
                    PositivePixels = np.sum(outimg)
                    PositivePercentage = 100 - ((PositivePixels / ImageArea) * 100)
                    ShownImagePositivePercentage.append(PositivePercentage)
                    outimg = Image.fromarray(outimg)
                    CanvasSize = 230, 230
                    outimg.thumbnail(CanvasSize, Image.ANTIALIAS)
                    PreviewImageMask = ImageTk.PhotoImage(outimg)
                    ShownImageThumbnailMasks.append(PreviewImageMask)

                print(ShownImagePositivePercentage)
                TotalWidth = 0
                CanvasXLoc = 0
                for I in ShownImagesThumbnailsSizes:
                    width = I[0]
                    TotalWidth += width

                DeadSpace = 1000 - TotalWidth
                DeadspaceFraction = round(DeadSpace / 3)
                for I in range(len(ShownImageThumbnailMasks)):
                    Nigel = ShownImageThumbnailMasks[I]
                    width, height = ShownImagesThumbnailsSizes[I]

                    print(width, height)
                    MaskCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
                    MaskCanvas.create_text(CanvasXLoc + 70, 290,
                                           text=f"Tissue: {str(round(ShownImagePositivePercentage[I], 2))}%",
                                           anchor="w", font=Medium_Font)
                    CanvasXLoc += (width + DeadspaceFraction)

            elif TypeDropdown.get() == "Adaptive Threshold - Mean":
                BlockSize = int(Block_Size_Entry.get())
                Blur = int(BlurEntry.get())
                Anchor = float(Anchor_Entry.get())

                print(Blur)

                for I in ShownImagesPaths:
                    img = cv2.imread(I, 0)
                    aug = iaa.AverageBlur(k=(Blur, Blur))
                    img = aug.augment_image(img)

                    outimg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BlockSize,
                                                   Anchor)
                    width, height = outimg.shape
                    ImageArea = (width * height) * 255
                    PositivePixels = np.sum(outimg)
                    PositivePercentage = 100 - ((PositivePixels / ImageArea) * 100)
                    ShownImagePositivePercentage.append(PositivePercentage)
                    outimg = Image.fromarray(outimg)
                    CanvasSize = 230, 230
                    outimg.thumbnail(CanvasSize, Image.ANTIALIAS)
                    PreviewImageMask = ImageTk.PhotoImage(outimg)
                    ShownImageThumbnailMasks.append(PreviewImageMask)

                print(ShownImagePositivePercentage)
                TotalWidth = 0
                CanvasXLoc = 0
                for I in ShownImagesThumbnailsSizes:
                    width = I[0]
                    TotalWidth += width

                DeadSpace = 1000 - TotalWidth
                DeadspaceFraction = round(DeadSpace / 3)
                for I in range(len(ShownImageThumbnailMasks)):
                    Nigel = ShownImageThumbnailMasks[I]
                    width, height = ShownImagesThumbnailsSizes[I]

                    print(width, height)
                    MaskCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
                    MaskCanvas.create_text(CanvasXLoc + 70, 290,
                                           text=f"Tissue: {str(round(ShownImagePositivePercentage[I], 2))}%",
                                           anchor="w", font=Medium_Font)
                    CanvasXLoc += (width + DeadspaceFraction)

            elif TypeDropdown.get() == "Adaptive Threshold - Gaussian":
                BlockSize = int(Block_Size_Entry.get())
                Blur = int(BlurEntry.get())
                Anchor = float(Anchor_Entry.get())

                print(Blur)

                for I in ShownImagesPaths:
                    img = cv2.imread(I, 0)
                    aug = iaa.AverageBlur(k=(Blur, Blur))
                    img = aug.augment_image(img)

                    outimg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                   BlockSize,
                                                   Anchor)
                    width, height = outimg.shape
                    ImageArea = (width * height) * 255
                    PositivePixels = np.sum(outimg)
                    PositivePercentage = 100 - ((PositivePixels / ImageArea) * 100)
                    ShownImagePositivePercentage.append(PositivePercentage)
                    outimg = Image.fromarray(outimg)
                    CanvasSize = 230, 230
                    outimg.thumbnail(CanvasSize, Image.ANTIALIAS)
                    PreviewImageMask = ImageTk.PhotoImage(outimg)
                    ShownImageThumbnailMasks.append(PreviewImageMask)

                print(ShownImagePositivePercentage)
                TotalWidth = 0
                CanvasXLoc = 0
                for I in ShownImagesThumbnailsSizes:
                    width = I[0]
                    TotalWidth += width

                DeadSpace = 1000 - TotalWidth
                DeadspaceFraction = round(DeadSpace / 3)
                for I in range(len(ShownImageThumbnailMasks)):
                    Nigel = ShownImageThumbnailMasks[I]
                    width, height = ShownImagesThumbnailsSizes[I]

                    print(width, height)
                    MaskCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
                    MaskCanvas.create_text(CanvasXLoc + 70, 290,
                                           text=f"Tissue: {str(round(ShownImagePositivePercentage[I], 2))}%",
                                           anchor="w", font=Medium_Font)
                    CanvasXLoc += (width + DeadspaceFraction)

            elif TypeDropdown.get() == "Otsu Binarisation":

                Blur = int(BlurEntry.get())

                for I in ShownImagesPaths:
                    img = cv2.imread(I, 0)
                    aug = iaa.AverageBlur(k=(Blur, Blur))
                    img = aug.augment_image(img)

                    ret, outimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    width, height = outimg.shape
                    ImageArea = (width * height) * 255
                    PositivePixels = np.sum(outimg)
                    PositivePercentage = 100 - ((PositivePixels / ImageArea) * 100)
                    ShownImagePositivePercentage.append(PositivePercentage)
                    outimg = Image.fromarray(outimg)
                    CanvasSize = 230, 230
                    outimg.thumbnail(CanvasSize, Image.ANTIALIAS)
                    PreviewImageMask = ImageTk.PhotoImage(outimg)
                    ShownImageThumbnailMasks.append(PreviewImageMask)

                print(ShownImagePositivePercentage)
                TotalWidth = 0
                CanvasXLoc = 0
                for I in ShownImagesThumbnailsSizes:
                    width = I[0]
                    TotalWidth += width

                DeadSpace = 1000 - TotalWidth
                DeadspaceFraction = round(DeadSpace / 3)
                for I in range(len(ShownImageThumbnailMasks)):
                    Nigel = ShownImageThumbnailMasks[I]
                    width, height = ShownImagesThumbnailsSizes[I]

                    MaskCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
                    MaskCanvas.create_text(CanvasXLoc + 70, 290,
                                           text=f"Tissue: {str(round(ShownImagePositivePercentage[I], 2))}%",
                                           anchor="w", font=Medium_Font)
                    CanvasXLoc += (width + DeadspaceFraction)

    def NextPreview():
        nonlocal ShownImageLocation

        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        ShownImageThumbnailMasks.clear()
        ShownImagePositivePercentage.clear()
        ShownImageLocation += 4

        if ShownImageLocation >= len(ImagePaths):
            ShownImageLocation = len(ImagePaths)

        First_Image = ShownImageLocation - 4

        if First_Image <= 0:
            First_Image = 0

        for x in ImagePaths[First_Image:ShownImageLocation]:
            ShownImagesPaths.append(x)

        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            print(type(Im))
            CanvasSize = 230, 230
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            print(ThumbnailSize)

            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)

        CanvasXLoc = 0

        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 1000 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 3)
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            print(width, height)
            OrigCanvas.create_image(CanvasXLoc, 130, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

        try:
            CreateMask()
        except:
            pass

        #################################################### Place Widgets #################################################

    def PrevPreview():

        nonlocal ShownImageLocation

        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        ShownImageThumbnailMasks.clear()
        ShownImagePositivePercentage.clear()
        ShownImageLocation -= 4

        if ShownImageLocation <= 4:
            ShownImageLocation = 4
        if ShownImageLocation >= len(ImagePaths):
            ShownImageLocation = len(ImagePaths)

        First_Image = ShownImageLocation - 4

        if First_Image <= 0:
            First_Image = 0

        for x in ImagePaths[First_Image:ShownImageLocation]:
            ShownImagesPaths.append(x)

        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            print(type(Im))
            CanvasSize = 230, 230
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            print(ThumbnailSize)

            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)

        CanvasXLoc = 0

        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 1000 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 3)
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            print(width, height)
            OrigCanvas.create_image(CanvasXLoc, 130, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

        try:
            CreateMask()
        except:
            pass

    def MaskFolderStructure():
        OriginalTopFolder = Step1Entry.get()
        MaskFolder = os.path.join(OriginalTopFolder, "Masks")
        Folders = []

        for r, d, f in os.walk(OriginalTopFolder):
            for file in f:
                if r not in Folders:
                    Folders.append(r)

        if not os.path.isdir(MaskFolder):
            os.mkdir(MaskFolder)

        for F in Folders:
            Short = F.replace(OriginalTopFolder,"")
            NewFolder = MaskFolder + Short

            if not os.path.isdir(NewFolder):
                os.makedirs(NewFolder)

    def WS_Thread_Threading():
        Errors = PreChecks()
        if Errors == 0:
            if KeepCheckVar.get() == 1:
                try:
                    NewFolder = os.path.join(Step1Entry.get(), "Removed Images")
                    os.mkdir(NewFolder)
                except:
                    pass

            if MaskCheckVar.get() == 1:
                MaskFolderStructure()

            t = threading.Thread(target=WS_Threading)
            t.start()
            ProgressPopup(len(ImagePaths))

    def WS_Threading():

        pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
        for I in range(len(ImagePaths)):
            pool.submit(WS_Core, ImagePaths[I])
        pool.shutdown(wait=True)

    def WS_Core(ImageIn):
        global Images_Processed
        # print(ImageIn)
        Blur = int(BlurEntry.get())
        if TypeDropdown.get() == "Positive Pixel Classification":
            Cutoff = float(Binary_Option_Entry.get())
            img = cv2.imread(ImageIn, 0)
            aug = iaa.AverageBlur(k=(Blur, Blur))
            img = aug.augment_image(img)
            ret, outimg = cv2.threshold(img, round(Cutoff * 255), 255, cv2.THRESH_BINARY)
            width, height = outimg.shape
            ImageArea = (width * height) * 255
            PositivePixels = np.sum(outimg)
            PositivePercentage = float(100 - ((PositivePixels / ImageArea) * 100))



            print(PositivePercentage, int(Step2Entry.get()))
            if PositivePercentage < int(Step2Entry.get()):
                if KeepCheckVar.get() == 1:
                    imagename = os.path.split(ImageIn)[-1]
                    DeletedLocation = os.path.join(Step1Entry.get(), "Removed Images", )
                    NewPath = os.path.join(DeletedLocation, imagename)

                    os.replace(ImageIn, NewPath)
                elif KeepCheckVar.get() == 0:
                    os.remove(ImageIn)

            elif MaskCheckVar.get() == 1:
                print("Making Masks")
                Img = Image.fromarray(outimg)
                File = os.path.split(ImageIn)[-1]
                Short = ImageIn.replace(Step1Entry.get(),"")
                MaskFolder = os.path.join(Step1Entry.get(), "Masks")
                NewFilePath = MaskFolder + Short
                print(NewFilePath)
                Img.save(NewFilePath)


        elif TypeDropdown.get() == "Adaptive Threshold - Mean":
            Block = int(Block_Size_Entry.get())
            Anchor = float(Anchor_Entry.get())

            img = cv2.imread(ImageIn, 0)
            aug = iaa.AverageBlur(k=(Blur, Blur))
            img = aug.augment_image(img)

            outimg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, Block, Anchor)
            width, height = outimg.shape
            ImageArea = (width * height) * 255
            PositivePixels = np.sum(outimg)
            PositivePercentage = 100 - ((PositivePixels / ImageArea) * 100)
            ShownImagePositivePercentage.append(PositivePercentage)




            if PositivePercentage < int(Step2Entry.get()):
                if KeepCheckVar.get() == 1:
                    imagename = os.path.split(ImageIn)[-1]
                    DeletedLocation = os.path.join(Step1Entry.get(), "Removed Images", )
                    NewPath = os.path.join(DeletedLocation, imagename)
                    os.replace(ImageIn, NewPath)
                elif KeepCheckVar.get() == 0:
                    os.remove(ImageIn)

            elif MaskCheckVar.get() == 1:
                print("Making Masks")
                Img = Image.fromarray(outimg)
                File = os.path.split(ImageIn)[-1]
                Short = ImageIn.replace(Step1Entry.get(),"")
                MaskFolder = os.path.join(Step1Entry.get(), "Masks")
                NewFilePath = MaskFolder + Short
                print(NewFilePath)
                Img.save(NewFilePath)

        elif TypeDropdown.get() == "Adaptive Threshold - Gaussian":
            Block = int(Block_Size_Entry.get())
            Anchor = float(Anchor_Entry.get())

            img = cv2.imread(ImageIn, 0)
            aug = iaa.AverageBlur(k=(Blur, Blur))
            img = aug.augment_image(img)

            outimg = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, Block, Anchor)
            width, height = outimg.shape
            ImageArea = (width * height) * 255
            PositivePixels = np.sum(outimg)
            PositivePercentage = 100 - ((PositivePixels / ImageArea) * 100)
            ShownImagePositivePercentage.append(PositivePercentage)

            if PositivePercentage < int(Step2Entry.get()):
                if KeepCheckVar.get() == 1:
                    imagename = os.path.split(ImageIn)[-1]
                    DeletedLocation = os.path.join(Step1Entry.get(), "Removed Images", )
                    NewPath = os.path.join(DeletedLocation, imagename)
                    os.replace(ImageIn, NewPath)
                elif KeepCheckVar.get() == 0:
                    os.remove(ImageIn)

            elif MaskCheckVar.get() == 1:
                print("Making Masks")
                Img = Image.fromarray(outimg)
                File = os.path.split(ImageIn)[-1]
                Short = ImageIn.replace(Step1Entry.get(),"")
                MaskFolder = os.path.join(Step1Entry.get(), "Masks")
                NewFilePath = MaskFolder + Short
                print(NewFilePath)
                Img.save(NewFilePath)

        elif TypeDropdown.get() == "Otsu Binarisation":
            img = cv2.imread(ImageIn, 0)
            aug = iaa.AverageBlur(k=(Blur, Blur))
            img = aug.augment_image(img)

            ret, outimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            width, height = outimg.shape
            ImageArea = (width * height) * 255
            PositivePixels = np.sum(outimg)
            PositivePercentage = 100 - ((PositivePixels / ImageArea) * 100)
            ShownImagePositivePercentage.append(PositivePercentage)

            if PositivePercentage < int(Step2Entry.get()):
                if KeepCheckVar.get() == 1:
                    imagename = os.path.split(ImageIn)[-1]
                    DeletedLocation = os.path.join(Step1Entry.get(), "Removed Images", )
                    NewPath = os.path.join(DeletedLocation, imagename)
                    os.replace(ImageIn, NewPath)
                elif KeepCheckVar.get() == 0:
                    os.remove(ImageIn)

            elif MaskCheckVar.get() == 1:
                print("Making Masks")
                Img = Image.fromarray(outimg)
                File = os.path.split(ImageIn)[-1]
                Short = ImageIn.replace(Step1Entry.get(),"")
                MaskFolder = os.path.join(Step1Entry.get(), "Masks")
                NewFilePath = MaskFolder + Short
                print(NewFilePath)
                Img.save(NewFilePath)

        Images_Processed += 1

    Step1Text = tk.Label(MainFrame, fg="black",
                         text="Select your Image Folder:", font=Medium_Font, bg="gray83")
    Step1Text.place(anchor="center", relx=0.17, rely=0.05)

    Step1Entry = tk.Entry(MainFrame, width=25, font=Large_Font)
    Step1Entry.place(anchor="center", relx=0.15, rely=0.1)

    ButtonForFiles2 = tk.Button(MainFrame, text="Browse...", font=Small_Font, command=LoadImages)
    ButtonForFiles2.place(anchor="center", relx=0.27, rely=0.1)

    Step2Text = tk.Label(MainFrame, fg="black", text="Select your minimum wanted tissue coverage. (0-100%)", font=Medium_Font, bg="gray83")
    Step2Text.place(anchor="center", relx=0.17, rely=0.15)

    Validation = (MainFrame.register(IntOnly))
    EV = tk.StringVar()
    Step2Entry = tk.Entry(MainFrame, width=5, textvariable=EV, justify="center", validate="key",
                          validatecommand=(Validation, "%P", "%d"))
    Step2Entry.place(anchor="center", relx=0.17, rely=0.2)
    Step2Entry.bind_all('<Key>', Change_Slider)

    Step2Scale = tk.Scale(MainFrame, from_=0, to=100, orient="horizontal", showvalue=0, tickinterval=50, bg="gray83", bd=0, highlightthickness=0, command=Change_Tissue_Entry)
    Step2Scale.place(anchor="center", relx=0.17, rely=0.25)

    Step3Label = tk.Label(MainFrame, text="Select your Image Thresholding Method", bg="gray83",font=Medium_Font)
    Step3Label.place(anchor="center", relx=0.17, rely=0.3)

    HelpLabel = tk.Label(MainFrame, text="", bg="gray83")
    HelpLabel.place(anchor="center", relx=0.17, rely=0.55)

    OptionList = ["Positive Pixel Classification", "Adaptive Threshold - Mean", "Adaptive Threshold - Gaussian",
                  "Otsu Binarisation"]
    TypeDropdown = ttk.Combobox(MainFrame, values=OptionList, width=35, state="readonly")
    TypeDropdown.place(relx=0.17, rely=0.35, anchor="center")
    TypeDropdown.bind("<<ComboboxSelected>>", lambda e: BringUpOptions())
    TypeDropdown.set("Positive Pixel Classification")


    Binary_Option_Label = tk.Label(MainFrame, text="Pixel Cutoff (0-1)", bg="gray83")
    Binary_Option_Entry = tk.Entry(MainFrame, width=4, justify="center")
    Binary_Option_Text = "The intensity of a pixel considered to be negative. A higher value includes lighter pixels."
    Binary_Option_Label.bind("<Enter>", lambda e: HelpLabel.config(text=Binary_Option_Text))
    Binary_Option_Label.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    Binary_Option_Label.place(relx=0.12, rely=0.4, anchor="center")
    Binary_Option_Entry.place(relx=0.20, rely=0.4, anchor="center")

    BlurLabel = tk.Label(MainFrame, text="Smoothing Intensity (Odd, >=0)", bg="gray83")
    BlurEntry = tk.Entry(MainFrame, width=4, justify="center", validate="key",
                          validatecommand=(Validation, "%P", "%d"))
    Blur_Option_Text = "Intensity of blur to add to the image. This can help fill in small gaps in the image"
    BlurLabel.bind("<Enter>", lambda e: HelpLabel.config(text=Blur_Option_Text))
    BlurLabel.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    BlurLabel.place(relx=0.12, rely=0.45, anchor="center")
    BlurEntry.place(relx=0.20, rely=0.45, anchor="center")

    Info_Label = tk.Label(MainFrame, text="Pixel classed as positive or negative based on a single threshold.\n"
                                             "The higher the threshold, the lighter the pixels included.", bg="gray83")
    Info_Label.place(anchor="c", relx=0.17, rely=0.6)

    Block_Size_Label = tk.Label(MainFrame, text="Block Size (Odd, >0)", bg="gray83")
    Block_Size_Entry = tk.Entry(MainFrame, width=4, justify="center", validate="key",
                          validatecommand=(Validation, "%P", "%d"))
    Block_Info = "Kernel size for the image. Must be an odd positive integer"
    Block_Size_Label.bind("<Enter>", lambda e: HelpLabel.config(text=Block_Info))
    Block_Size_Label.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    Anchor_Label = tk.Label(MainFrame, text="Anchor Value (>1)", bg="gray83")
    Anchor_Entry = tk.Entry(MainFrame, width=4, justify="center", validate="key",
                          validatecommand=(Validation, "%P", "%d"))
    Anchor_Info = "The value subtracted from the mean or weighed mean calculated. Must be a positive integer"
    Anchor_Label.bind("<Enter>", lambda e: HelpLabel.config(text=Anchor_Info))
    Anchor_Label.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    Preview_Button = tk.Button(MainFrame, text="Preview", command=CreateMask)
    Preview_Button.place(anchor="center", relx=0.17, rely=0.5)

    KeepCheckVar = tk.IntVar(value=1)
    KeepChecker = tk.Checkbutton(MainFrame, variable=KeepCheckVar, offvalue=0, onvalue=1, text="Keep sub-threshold images in separate folder?", bg="gray83")
    KeepChecker.place(anchor="center", relx=0.17, rely=0.725)

    MaskCheckVar = tk.IntVar(value=1)
    MaskChecker = tk.Checkbutton(MainFrame, variable=MaskCheckVar, offvalue=0, onvalue=1, text="Create Binary Masks of Remaining Images", bg="gray83")
    MaskChecker.place(anchor="center", relx=0.17, rely=0.775)

    OrigLabel = tk.Label(MainFrame, text="Original Images", font=Large_Font, bg="gray83")
    OrigLabel.place(anchor="center", relx=0.65, rely=0.025)
    OrigCanvas = tk.Canvas(MainFrame, height=260, width=1000)
    OrigCanvas.place(anchor="n", relx=0.65, rely=0.08)

    MaskLabel = tk.Label(MainFrame, text="Tissue Masks", font=Large_Font, bg="gray83")
    MaskLabel.place(anchor="center", relx=0.65, rely=0.42)
    MaskCanvas = tk.Canvas(MainFrame, height=260, width=1000)
    MaskCanvas.place(anchor="n", relx=0.65, rely=0.45)

    Prev_Img_Button = tk.Button(MainFrame, text="Previous Set", width=11, command=PrevPreview)
    Prev_Img_Button.place(anchor="center", relx=0.6, rely=0.85)

    Next_Img_Button = tk.Button(MainFrame, text="Next Set", width=11, command=NextPreview)
    Next_Img_Button.place(anchor="center", relx=0.7, rely=0.85)

    BeginButton = tk.Button(MainFrame, text="Start", height=1, width=15, command=WS_Thread_Threading)
    BeginButton.place(anchor="center", relx=0.65, rely=0.9)

def BalanceModule():
    global Active_Module
    Active_Module = "Balance"

    SetList = []
    LabelList = []
    BrowseButtons = []
    SetEntries = []
    CountList = []
    ImageListOfLists = []
    FolderLocations = []
    BalanceFolderLocations = []
    BalanceImageNames = []

    SizeList = []
    DifferenceList = []
    ImagesToAdd = []
    ImagesToRemove = []

    BeenAugmentedPath = []
    BeenAugmentedAugments = []



    SetOptions = 2 # Number of Sets currently active in the balance canvas
    BottomEdge = 210 # Bottom edge of the last image set frame in pixels
    Images_To_Process = 0

    # Binds Scroll wheel to canvas
    def Scrolly(e):
        SetCanvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def AddOption():
        nonlocal SetOptions
        nonlocal BottomEdge
        TopEdge = BottomEdge + 20
        #print(BottomEdge)

        SetOptions += 1

        SetFrameInner = tk.Frame(SetCanvas, height=100, width=350, relief="groove", borderwidth=1)

        SetLabel = tk.Label(SetFrameInner, text=f"Image Set {SetOptions}", name=f"label{SetOptions}")
        SetLabel.place(relx=0.5, rely=0.2, anchor="center")
        LabelList.append(SetLabel)

        FolderEntry = tk.Entry(SetFrameInner, width=40)
        FolderEntry.place(anchor="center", relx=0.4, rely=0.5)
        SetEntries.append(FolderEntry)

        ButtonForFiles = tk.Button(SetFrameInner, text="Browse...", name=f"button{SetOptions}", command=lambda: NameSet(str(ButtonForFiles)))
        ButtonForFiles.place(anchor="center", relx=0.85, rely=0.5)
        BrowseButtons.append(ButtonForFiles)

        Count = tk.Label(SetFrameInner, text="", fg="blue")
        Count.place(anchor="center", relx=0.5, rely=0.8)
        CountList.append(Count)

        ID = SetCanvas.create_window(200, TopEdge, window=SetFrameInner, anchor="n")
        SetList.append(ID)

        BottomEdge += 110

        if BottomEdge > 250:
            SetCanvas.config(scrollregion=(0, 0, 400, BottomEdge + 20))

    def RemoveOption():
        nonlocal SetOptions
        nonlocal BottomEdge
        print(SetList[-1])
        if len(SetList) > 2:
            SetCanvas.delete(SetList[-1])
            del SetList[-1]
            del LabelList[-1]
            del BrowseButtons[-1]
            del SetEntries[-1]
            del CountList[-1]
            try:
                del ImageListOfLists[-1]
            except:
                pass
            try:
                del FolderLocations[-1]
            except:
                pass

        print(ImageListOfLists)
        if SetOptions > 2:
            SetOptions -= 1

        if BottomEdge > 220:
            BottomEdge -= 110

        if BottomEdge > 250:
            SetCanvas.config(scrollregion=(0, 0, 400, BottomEdge + 20))
        else:
            SetCanvas.config(scrollregion=(0, 0, 400, 250))

        print(BottomEdge)

        SizeList = []
        for x in ImageListOfLists:
            SizeList.append(len(x))
        print(SizeList)
        Smallest = min(SizeList)
        Biggest = max(SizeList)

        SmallestLoc = SizeList.index(Smallest) + 1
        BiggestLoc = SizeList.index(Biggest) + 1

        if len(SizeList) > 1:
            result = all(elem == SizeList[0] for elem in SizeList)
            if result:
                MinMaxLabel.config(text="Image count in all sets are equal")
            else:
                MinMaxLabel.config(text=f"Least Images: {Smallest} in Image Set {SmallestLoc}\n"
                                        f"Most Images: {Biggest} in Image Set {BiggestLoc}\n"
                                        f"Average Images: {round(sum(SizeList) / len(SizeList))}")

                BalanceInfo()

    def NameSet(ButtonNumber):
        ImageList = []

        OptionsNumber = int(ButtonNumber[-1]) - 1
        try:
            ImageListOfLists.pop(OptionsNumber)
            FolderLocations.pop(OptionsNumber)
        except:
            print("Not Present")

        SelectedEntry = SetEntries[OptionsNumber]
        #print(SelectedEntry)
        SelectedEntry.delete(0, 10000)
        Directory = filedialog.askdirectory(initialdir=os.getcwd(), title="Please select your top-level folder.")
        SelectedEntry.insert(0, Directory)
        for r, d, f in os.walk(SelectedEntry.get()):
            for file in f:
                if '.jpg' in file or '.jfif' in file or ".png" in file:
                    ImageList.append(os.path.join(r, file))
        CountList[OptionsNumber].config(text=f"{len(ImageList)} Images Detected")
        # print(ImageList)
        FolderLocations.insert(OptionsNumber, SelectedEntry.get())
        ImageListOfLists.insert(OptionsNumber, ImageList)
        #print(ImageListOfLists)
        #print(len(ImageListOfLists))
        #print(FolderLocations)
        #print(len(FolderLocations))

        if os.path.isdir(SelectedEntry.get()):
            LabelList[OptionsNumber].config(text=f"Image Set {ButtonNumber[-1]}", fg="black")

        SizeList = []
        for x in ImageListOfLists:
            SizeList.append(len(x))

        print(SizeList)
        Smallest = min(SizeList)
        Biggest = max(SizeList)

        SmallestLoc = SizeList.index(Smallest) + 1
        BiggestLoc = SizeList.index(Biggest) + 1

        if len(SizeList) > 1:
            result = all(elem == SizeList[0] for elem in SizeList)
            if result:
                MinMaxLabel.config(text="Image count in all sets are equal")
            else:
                MinMaxLabel.config(text=f"Least Images: {Smallest} in Image Set {SmallestLoc}\n"
                                        f"Most Images: {Biggest} in Image Set {BiggestLoc}\n"
                                        f"Average Images: {round(sum(SizeList) / len(SizeList))}")
                BalanceInfo()
        else:
            pass

    def BalanceInfo():
        SizeList = []
        for x in ImageListOfLists:
            SizeList.append(len(x))

        # print(SizeList)
        Smallest = min(SizeList)
        Biggest = max(SizeList)

        if VarBal.get() == 0:
            Update_Label.config(text=f"Image Sets will be balanced to {Biggest} images")
        elif VarBal.get() == 1:
            Update_Label.config(text=f"Image Sets will be balanced to {Smallest} images")
        elif VarBal.get() == 2:
            Update_Label.config(text=f"Image Sets will be balanced to {round(sum(SizeList) / len(SizeList))} images")

    def PreChecks():
        Errors = 0
        for X in range(len(SetEntries)):
            if len(SetEntries[X].get()) == 0:
                LabelList[X].config(fg="red", text="Please enter a valid directory")
                Errors += 1
            else:
                LabelList[X].config(fg="black", text=f"Image Set {X + 1}")

            if not os.path.isdir(SetEntries[X].get()):
                LabelList[X].config(fg="red", text="Please enter a valid directory")
                Errors += 1
            else:
                LabelList[X].config(fg="black", text=f"Image Set {X + 1}")

        if Errors != 0:
            pass
        else:
            pass

        return Errors

    def MakeBalanceLists():
        nonlocal SizeList
        WarningNeeded = False

        print(DifferenceList)
        print(f"SizeListLength: {len(SizeList)}")
        for Y in range(len(SizeList)):
            print(f"Processing set {Y+1}")
            UniqueID = 0
            ImageSet = ImageListOfLists[Y]
            ImageRoot  = FolderLocations[Y]
            ValueToChange = DifferenceList[Y]
            print("Value to Change " + str(ValueToChange))
            print("ImageSet Value " + str(len(ImageSet)))

            if ValueToChange > 0:

                if ValueToChange > len(ImageSet):
                    print("More than double largest set; implementing counter measures")
                    TimesSizefloat = ValueToChange / len(ImageSet)
                    print(f"This set is {TimesSizefloat} smaller than the biggest set")
                    TimesSize = math.floor(ValueToChange / len(ImageSet))
                    if TimesSize > 7:
                        WarningNeeded = True
                    print("Timessize ", TimesSize)
                    print(int((TimesSizefloat % 1) * (float(len(ImageSet)))))
                    IMGTOADD = 0
                    Fraction = math.ceil((TimesSizefloat % 1) * len(ImageSet))

                    for T in range(TimesSize):
                        for i in range(len(ImageSet)):
                            Path_and_data = []
                            Path_and_data.append(ImageSet[i])
                            Path_and_data.append(ImageRoot)
                            if TimesSize > 7:
                                Path_and_data.append(">7")
                            else:
                                Path_and_data.append("2-7")

                            ImagesToAdd.append(Path_and_data)
                            UniqueID += 1
                            IMGTOADD += 1
                    print(f"Fraction: {Fraction}")
                    Images_For_Balancing = random.sample(ImageSet, k=Fraction)
                    for i in range(len(Images_For_Balancing)):
                        Path_and_data = []
                        Path_and_data.append(Images_For_Balancing[i])
                        Path_and_data.append(ImageRoot)
                        if TimesSize > 7:
                            Path_and_data.append(">7")
                        else:
                            Path_and_data.append("2-7")
                        UniqueID += 1
                        IMGTOADD += 1
                        ImagesToAdd.append(Path_and_data)
                    print(f"Total images to add for set:  {IMGTOADD}")

                elif ValueToChange <= len(ImageSet):

                    Images_For_Balancing = random.sample(ImageSet, k=ValueToChange)
                    for i in range(len(Images_For_Balancing)):
                        Path_and_data = []
                        Path_and_data.append(Images_For_Balancing[i])
                        Path_and_data.append(ImageRoot)
                        Path_and_data.append("<2")
                        ImagesToAdd.append(Path_and_data)

                        UniqueID += 1

            elif ValueToChange < 0:
                if abs(ValueToChange) > len(ImageSet):
                    Images_For_Balancing = random.choices(ImageSet, k=abs(ValueToChange))
                    for i in Images_For_Balancing:
                        ImagesToRemove.append(i)
                elif abs(ValueToChange) < len(ImageSet):
                    Images_For_Balancing = random.sample(ImageSet, k=abs(ValueToChange))
                    for i in Images_For_Balancing:
                        ImagesToRemove.append(i)

            else:
                pass
            print(f"Adding {UniqueID} to this class")
            print("------------------------------------------")
        print(f"SizeList: {SizeList}")
        print(f"DifferenceList: {DifferenceList}")
        print(f"Images to Remove: {len(ImagesToRemove)}")
        print(f"Images to Add:{len(ImagesToAdd)}")


        if WarningNeeded:
            WarningPopup()
        else:
            Balance_Thread_Threading()

    def RemoveImages(Image):
        global Images_Processed
        os.remove(Image)
        Images_Processed += 1
        #print(Images_Processed)

    def AddImages(Image):
        nonlocal BeenAugmentedPath
        global Images_Processed

        ImagePath = Image[0] # Path of the Image
        ImageRoot = Image[1] # Root Directory of the Image
        Scenario = Image[2] # How to deal with image

        def RollAugments():
            Rotation = [0, 90, 180, 270]
            Flip = [1, 2]
            Angle = random.choice(Rotation)
            FlipType = random.choice(Flip)
            AugDef = f"Flip{FlipType}Rot{Angle} "

            while Angle == 0 and FlipType == 2:
                Angle = random.choice(Rotation)
                FlipType = random.choice(Flip)
                AugDef = f"Flip{FlipType}Rot{Angle} "

            if FlipType == 1:
                flip = iaa.Fliplr(1)
                Augments = iaa.Sequential([iaa.Affine(rotate=Angle), flip])

            else:
                Augments = iaa.Affine(rotate=Angle)

            return Augments, AugDef



        Image_in = imageio.imread(ImagePath)
        ImageName = os.path.split(ImagePath)[-1]


        SeperateBalanceFolder = os.path.join(ImageRoot, "Balanced Images")
        NotSeperateBalanceFolder = os.path.split(ImagePath)[0]

        if Scenario == "<2":

            Augments, AugDef = RollAugments()

            Image_out = Augments.augment_image(Image_in)
            Image_Out_Name = AugDef + ImageName

            if FolderVal.get() == 2:
                 print("Mixing with Original Images")
                 Image_Out_Filepath = os.path.join(NotSeperateBalanceFolder, Image_Out_Name)
            else:
                Image_Out_Filepath = os.path.join(SeperateBalanceFolder, Image_Out_Name)

            if not os.path.isfile(Image_Out_Filepath):
                imageio.imwrite(Image_Out_Filepath, Image_out)
            else:
                print("File already exists, writing duplicate.")
                dupe = 1
                while os.path.isfile(Image_Out_Filepath):
                    Head = os.path.split(Image_Out_Filepath)[0]
                    Tail = os.path.split(Image_Out_Filepath)[-1]
                    FileName_NoEXT = Tail.split(".")[:-1]
                    EXT = Tail.split(".")[-1]
                    if dupe == 1:
                        DupeFilename = "".join(string for string in FileName_NoEXT) + f"({dupe})." + EXT
                        Image_Out_Filepath = os.path.join(Head, DupeFilename)
                    else:
                        Image_Out_Filepath = Image_Out_Filepath.replace(f"({dupe-1}).", f"({dupe}).")

                    #print(Image_Out_Filepath)
                    dupe += 1
                imageio.imwrite(Image_Out_Filepath, Image_out)
            Images_Processed += 1
            #print(Images_Processed)

        elif Scenario == "2-7":


            Augments, AugDef = RollAugments()

            Image_out = Augments.augment_image(Image_in)
            Image_Out_Name = AugDef + ImageName
            #print("-------------------------------------------------------------")
            ##print(Image_Out_Name)
            if FolderVal.get() == 2:
                Image_Out_Filepath = os.path.join(NotSeperateBalanceFolder, Image_Out_Name)
            else:
                Image_Out_Filepath = os.path.join(SeperateBalanceFolder, Image_Out_Name)

            if not os.path.isfile(Image_Out_Filepath): # Is Filename Available? -Yes
                #print("FileName Available, saving...")
                imageio.imwrite(Image_Out_Filepath, Image_out)
                if ImagePath not in BeenAugmentedPath:
                    BeenAugmentedPath.append(ImagePath)
                    BeenAugmentedAugments.append([AugDef])
                else:
                    ListLocation = BeenAugmentedPath.index(ImagePath)
                    BeenAugmentedAugments[ListLocation].append(AugDef)
            else:
                if ImagePath in BeenAugmentedPath: # If this image has been done before...
                    #print("Augment Already done on this image")
                    ListLocation = BeenAugmentedPath.index(ImagePath)
                    AugmentsToImage = BeenAugmentedAugments[ListLocation]

                    if len(AugmentsToImage) < 7: # If there are still augmentations available...
                        #print("Rerolling augmentation")
                        while AugDef in AugmentsToImage: # Get an augment that has not been used yet
                            Augments, AugDef = RollAugments()
                        Image_Out_Name = AugDef + ImageName


                        if FolderVal.get() == 2:
                            Image_Out_Filepath = os.path.join(NotSeperateBalanceFolder, Image_Out_Name)
                        else:
                            Image_Out_Filepath = os.path.join(SeperateBalanceFolder, Image_Out_Name)

                        if not os.path.isfile(Image_Out_Filepath):  # Is Filename Available Now? -Yes
                            Image_out = Augments.augment_image(Image_in)
                            imageio.imwrite(Image_Out_Filepath, Image_out)
                            BeenAugmentedAugments[ListLocation].append(AugDef)

                        else: # Is Filename Available Now? -No
                            #print("New Augment for this image found but filename taken. Rewriting filename")
                            dupe = 1
                            while os.path.isfile(Image_Out_Filepath): # Add the (1,2,3,4...) etc
                                Head = os.path.split(Image_Out_Filepath)[0]
                                Tail = os.path.split(Image_Out_Filepath)[-1]
                                Ext = Tail.split(".")[-1]
                                Image_Name = Tail.split(".")[:-1]
                                Image_Name = "".join(Image_Name)
                                if dupe == 1:
                                    New_Image_Name = Image_Name + f"({dupe})." + Ext
                                    Image_Out_Filepath = os.path.join(Head, New_Image_Name)
                                else:
                                    Image_Name.replace(f"({dupe-1}).",f"({dupe}).")
                                    Image_Out_Filepath = os.path.join(Head, Image_Name)

                            Image_out = Augments.augment_image(Image_in)
                            imageio.imwrite(Image_Out_Filepath, Image_out)
                            BeenAugmentedAugments[ListLocation].append(AugDef)

                    else:
                        print("All Augments Used for this image")

                else: # If this image Filepath is taken but it has not been done before
                   # print("Augment not done on this image, rewriting filename")
                    dupe = 1
                    while os.path.isfile(Image_Out_Filepath):  # Add the (1,2,3,4...) etc
                        Head = os.path.split(Image_Out_Filepath)[0]
                        Tail = os.path.split(Image_Out_Filepath)[-1]
                        Ext = Tail.split(".")[-1]
                        Image_Name = Tail.split(".")[:-1]
                        Image_Name = "".join(Image_Name)
                        if dupe == 1:
                            New_Image_Name = Image_Name + f"({dupe})." + Ext
                            Image_Out_Filepath = os.path.join(Head, New_Image_Name)
                        else:
                            Image_Name.replace(f"({dupe - 1}).", f"({dupe}).")
                            Image_Out_Filepath = os.path.join(Head, Image_Name)

                    Image_out = Augments.augment_image(Image_in)
                    imageio.imwrite(Image_Out_Filepath, Image_out)
                    BeenAugmentedPath.append(ImagePath)
                    BeenAugmentedAugments.append([AugDef])
           # print(BeenAugmentedPath)
           # print(BeenAugmentedAugments)


        elif Scenario == ">7":

            Augments, AugDef = RollAugments()

            Image_out = Augments.augment_image(Image_in)
            Image_Out_Name = AugDef + ImageName
            #print("-------------------------------------------------------------")
            #print(Image_Out_Name)
            if FolderVal.get() == 2:
                Image_Out_Filepath = os.path.join(NotSeperateBalanceFolder, Image_Out_Name)
            else:
                Image_Out_Filepath = os.path.join(SeperateBalanceFolder, Image_Out_Name)

            if not os.path.isfile(Image_Out_Filepath):  # Is Filename Available? -Yes
               # print("FileName Available, saving...")
                imageio.imwrite(Image_Out_Filepath, Image_out)
                if ImagePath not in BeenAugmentedPath:
                    BeenAugmentedPath.append(ImagePath)
                    BeenAugmentedAugments.append([AugDef])
                else:
                    ListLocation = BeenAugmentedPath.index(ImagePath)
                    BeenAugmentedAugments[ListLocation].append(AugDef)
            else:
                if ImagePath in BeenAugmentedPath:  # If this image has been done before...
                   # print("Augment Already done on this image")
                    ListLocation = BeenAugmentedPath.index(ImagePath)
                    AugmentsToImage = BeenAugmentedAugments[ListLocation]

                    if len(AugmentsToImage) < 7:  # If there are still augmentations available...
                       # print("Rerolling augmentation")
                        while AugDef in AugmentsToImage:  # Get an augment that has not been used yet
                            Augments, AugDef = RollAugments()
                        Image_Out_Name = AugDef + ImageName

                        if FolderVal.get() == 2:
                            Image_Out_Filepath = os.path.join(NotSeperateBalanceFolder, Image_Out_Name)
                        else:
                            Image_Out_Filepath = os.path.join(SeperateBalanceFolder, Image_Out_Name)

                        if not os.path.isfile(Image_Out_Filepath):  # Is Filename Available Now? -Yes
                            Image_out = Augments.augment_image(Image_in)
                            imageio.imwrite(Image_Out_Filepath, Image_out)
                            BeenAugmentedAugments[ListLocation].append(AugDef)

                        else:  # Is Filename Available Now? -No
                          #  print("New Augment for this image found but filename taken. Rewriting filename")
                            dupe = 1
                            while os.path.isfile(Image_Out_Filepath):  # Add the (1,2,3,4...) etc
                                Head = os.path.split(Image_Out_Filepath)[0]
                                Tail = os.path.split(Image_Out_Filepath)[-1]
                                Ext = Tail.split(".")[-1]
                                Image_Name = Tail.split(".")[:-1]
                                Image_Name = "".join(Image_Name)
                                if dupe == 1:
                                    New_Image_Name = Image_Name + f"({dupe})." + Ext
                                    Image_Out_Filepath = os.path.join(Head, New_Image_Name)
                                else:
                                    Image_Name.replace(f"({dupe - 1}).", f"({dupe}).")
                                    Image_Out_Filepath = os.path.join(Head, Image_Name)

                            Image_out = Augments.augment_image(Image_in)
                            imageio.imwrite(Image_Out_Filepath, Image_out)
                            BeenAugmentedAugments[ListLocation].append(AugDef)

                    else:
                        #print("All Augments Used for this image - Making a duplicate augmentation")

                        dupe = 1
                        while os.path.isfile(Image_Out_Filepath):  # Add the (1,2,3,4...) etc
                            Head = os.path.split(Image_Out_Filepath)[0]
                            Tail = os.path.split(Image_Out_Filepath)[-1]
                            Ext = Tail.split(".")[-1]
                            Image_Name = Tail.split(".")[:-1]
                            Image_Name = "".join(Image_Name)
                            if dupe == 1:
                                New_Image_Name = Image_Name + f"({dupe})." + Ext
                                Image_Out_Filepath = os.path.join(Head, New_Image_Name)
                            else:
                                Image_Name.replace(f"({dupe - 1}).", f"({dupe}).")
                                Image_Out_Filepath = os.path.join(Head, Image_Name)

                        Image_out = Augments.augment_image(Image_in)
                        imageio.imwrite(Image_Out_Filepath, Image_out)


                else:  # If this image Filepath is taken but it has not been done before
                   # print("Augment not done on this image, rewriting filename")
                    dupe = 1
                    while os.path.isfile(Image_Out_Filepath):  # Add the (1,2,3,4...) etc
                        Head = os.path.split(Image_Out_Filepath)[0]
                        Tail = os.path.split(Image_Out_Filepath)[-1]
                        Ext = Tail.split(".")[-1]
                        Image_Name = Tail.split(".")[:-1]
                        Image_Name = "".join(Image_Name)
                        if dupe == 1:
                            New_Image_Name = Image_Name + f"({dupe})." + Ext
                            Image_Out_Filepath = os.path.join(Head, New_Image_Name)
                        else:
                            Image_Name.replace(f"({dupe - 1}).", f"({dupe}).")
                            Image_Out_Filepath = os.path.join(Head, Image_Name)

                    Image_out = Augments.augment_image(Image_in)
                    imageio.imwrite(Image_Out_Filepath, Image_out)
                    BeenAugmentedPath.append(ImagePath)
                    BeenAugmentedAugments.append([AugDef])
            #print(BeenAugmentedPath)
            #print(BeenAugmentedAugments)

        Images_Processed += 1
        #print(Images_Processed)

    def AddBalanceFolders():

        if FolderVal.get() == 1:
            print(f"Folder Locations: {len(FolderLocations)}")
            for f in range(len(FolderLocations)):
                if DifferenceList[f] < 0:
                    BalancedFolder = os.path.join(FolderLocations[f], "Balanced Images")
                    print(BalancedFolder)
                    BalanceFolderLocations.append(BalancedFolder)
                elif DifferenceList[f] > 0:
                    BalancedFolder = os.path.join(FolderLocations[f], "Balanced Images")
                    print(BalancedFolder)
                    BalanceFolderLocations.append(BalancedFolder)
                    try:
                        os.mkdir(BalancedFolder)
                    except FileExistsError:
                        print("Folder already exists")
                else:
                    BalancedFolder = os.path.join(FolderLocations[f], "Balanced Images")
                    print(BalancedFolder)
                    BalanceFolderLocations.append(BalancedFolder)
        else:
            pass

    def WarningPopup():
        WarnPopup = tk.Toplevel(MainScreen)
        WarnPopup.geometry(f"500x100+{ScreenWidthMiddle - 250}+{ScreenHeightMiddle - 50}")
        WarnPopup.title("WARNING")
        WarnPopup.resizable(False, False)
        WarnPopup.iconbitmap(r"Icon/HistoSquare.ico")
        WarnPopup.grab_set()

        def Close():
            WarnPopup.grab_release()
            WarnPopup.destroy()

        def Continue():
            WarnPopup.grab_release()
            WarnPopup.destroy()
            AddBalanceFolders()
            Balance_Thread_Threading()


        Warnlabel = tk.Label(WarnPopup, text="One or more of your datasets are >7 times bigger than your smallest.\n"
                                             "This will lead to repeat augmentations.\n"
                                             "Do you wish to continue?")
        Warnlabel.place(relx=0.5, rely=0.25, anchor="center")

        ContinueButton = tk.Button(WarnPopup, text="Continue", command=Continue)
        ContinueButton.place(relx=0.3, rely=0.8, anchor="center")

        CancelButton = tk.Button(WarnPopup, text="Cancel", command=Close)
        CancelButton.place(relx=0.7, rely=0.8, anchor="center")

    def Adjust_ProgressBar():
        nonlocal Images_To_Process
        Images_To_Process = 0
        SizeList.clear()
        DifferenceList.clear()
        ImagesToAdd.clear()
        ImagesToRemove.clear()

        for x in ImageListOfLists:
            SizeList.append(len(x))

        Errors = PreChecks()

        if Errors == 0:
            DifferenceListAbs = []
            Smallest = min(SizeList)
            Biggest = max(SizeList)
            Average = round(sum(SizeList) / len(SizeList))

            if VarBal.get() == 0:
                Target = Biggest
            elif VarBal.get() == 1:
                Target = Smallest
            elif VarBal.get() == 2:
                Target = Average
            else:
                Target = 0

            for X in SizeList:
                Diffabs = abs(Target - X)
                Diff = Target - X
                DifferenceListAbs.append(Diffabs)
                DifferenceList.append(Diff)

            ImagesToProcess = sum(DifferenceListAbs)
            Images_To_Process += ImagesToProcess

            MakeBalanceLists()

            print(f"Images to Process {ImagesToProcess}")

    def Balance_Thread_Threading():
        nonlocal  Images_To_Process
        AddBalanceFolders()
        t = threading.Thread(target=Balance_Threading)
        t.start()
        ProgressPopup(Images_To_Process)

    def ReturnFolderOption():
        print(FolderVal.get())

    def Balance_Threading():
        print(f"Removing {len(ImagesToRemove)}")
        print(f"Adding {len(ImagesToAdd)}")
        Start = time.time()
        if len(ImagesToRemove) > 0:
            pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
            for I in range(len(ImagesToRemove)):
                pool.submit(RemoveImages, ImagesToRemove[I])
            pool.shutdown(wait=True)

        if len(ImagesToAdd) > 0:
            print("Made It Here")
            pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
            for I in range(len(ImagesToAdd)):
                #print(I)
                pool.submit(AddImages, ImagesToAdd[I])
            pool.shutdown(wait=True)
        End = time.time()
        print(f"Time Elapsed: {Start-End}")

    UppText = tk.Label(MainFrame, fg="black", text=f"Here you can quick-balance your data set.", font=Large_Font, bg="gray83")
    UppText.place(anchor="center", relx=0.5, rely=0.05)

    SetFrameOutter = tk.Frame(MainFrame, width=400, height=350, relief="groove")
    SetFrameOutter.place(anchor="center", relx=0.5, rely=0.35)
    SetCanvas = tk.Canvas(SetFrameOutter, width=400, height=400, bg="gray86", scrollregion=(0, 0, 400, 400),
                          highlightthickness=4)
    SetScroll = tk.Scrollbar(SetFrameOutter, orient="vertical", command=SetCanvas.yview)
    SetScroll.pack(side="left", fill="y")
    SetCanvas.config(yscrollcommand=SetScroll.set)
    SetCanvas.pack(side="left", expand=True, fill="both")
    SetCanvas.bind_all("<MouseWheel>", Scrolly)

    SetFrameInner1 = tk.Frame(SetCanvas, height=100, width=350, relief="groove", borderwidth=1)

    SetLabel1 = tk.Label(SetFrameInner1, text=f"Image Set 1", name=f"label1")
    SetLabel1.place(relx=0.5, rely=0.2, anchor="center")
    LabelList.append(SetLabel1)

    Folder1Entry = tk.Entry(SetFrameInner1, width=40)
    Folder1Entry.place(anchor="center", relx=0.4, rely=0.5)
    SetEntries.append(Folder1Entry)

    ButtonForFiles1 = tk.Button(SetFrameInner1, text="Browse...", command=lambda: NameSet(str(1)))
    ButtonForFiles1.place(anchor="center", relx=0.85, rely=0.5)
    BrowseButtons.append(ButtonForFiles1)

    Count1 = tk.Label(SetFrameInner1, text="", fg="blue")
    Count1.place(anchor="center", relx=0.5, rely=0.8)
    CountList.append(Count1)

    ID1 = SetCanvas.create_window(200, 10, window=SetFrameInner1, anchor="n")
    SetList.append(ID1)

    SetFrameInner2 = tk.Frame(SetCanvas, height=100, width=350, relief="groove", borderwidth=1)

    SetLabel2 = tk.Label(SetFrameInner2, text=f"Image Set 2", name=f"label2")
    SetLabel2.place(relx=0.5, rely=0.2, anchor="center")
    LabelList.append(SetLabel2)

    Folder2Entry = tk.Entry(SetFrameInner2, width=40, name="entry2")
    Folder2Entry.place(anchor="center", relx=0.4, rely=0.5)
    SetEntries.append(Folder2Entry)

    ButtonForFiles2 = tk.Button(SetFrameInner2, text="Browse...", command=lambda: NameSet(str(2)))
    ButtonForFiles2.place(anchor="center", relx=0.85, rely=0.5)
    BrowseButtons.append(ButtonForFiles2)

    Count2 = tk.Label(SetFrameInner2, text="", fg="blue")
    Count2.place(anchor="center", relx=0.5, rely=0.8)
    CountList.append(Count2)

    ID2 = SetCanvas.create_window(200, 120, window=SetFrameInner2, anchor="n")
    SetList.append(ID2)

    Add_Button = tk.Button(MainFrame, text="Add Set", width=10, command=AddOption)
    Add_Button.place(anchor="center", relx=0.45, rely=0.63)
    Remove_Button = tk.Button(MainFrame, text="Remove Set", command=RemoveOption)
    Remove_Button.place(anchor="center", relx=0.55, rely=0.63)

    VarBal = tk.IntVar()
    Bal_Up = tk.Radiobutton(MainFrame, text="Balance Up", variable=VarBal, value=0, bg="gray83", command=BalanceInfo)
    Bal_Up.place(anchor="center", relx=0.4, rely=0.78)
    Bal_Down = tk.Radiobutton(MainFrame, text="Balance Down", variable=VarBal, value=1, bg="gray83", command=BalanceInfo)
    Bal_Down.place(anchor="center", relx=0.5, rely=0.78)
    Bal_Avg = tk.Radiobutton(MainFrame, text="Balance to Average", variable=VarBal, value=2, bg="gray83", command=BalanceInfo)
    Bal_Avg.place(anchor="center", relx=0.6, rely=0.78)

    MinMaxLabel = tk.Label(MainFrame, text="", fg="blue", bg="gray83")
    MinMaxLabel.place(anchor="center", relx=0.5, rely=0.71)

    FolderVal = tk.IntVar(value=1)
    SingleFolder = tk.Radiobutton(MainFrame, text="Seperate Folder for New Images", variable=FolderVal, value=1, bg="gray83", command=ReturnFolderOption)
    SingleFolder.place(anchor="center", relx=0.4, rely=0.87)
    OrigFolders = tk.Radiobutton(MainFrame, text="Mix New Images with Original Images", variable=FolderVal, value=2, bg="gray83", command=ReturnFolderOption)
    OrigFolders.place(anchor="center", relx=0.6, rely=0.87)
    BeginButtonB = tk.Button(MainFrame, text="Begin Balancing", height=1, width=15, command=Adjust_ProgressBar)
    BeginButtonB.place(anchor="center", relx=0.5, rely=0.93)

    Update_Label = tk.Label(MainFrame, fg="blue", text="", bg="gray83")
    Update_Label.place(anchor="center", relx=0.5, rely=0.82)

def NormalisationModule():
    global Active_Module
    Active_Module = "Norm"
    TargetImage = []
    ImagePaths = []
    ShownImagesPaths = []
    ShownImagesThumbnails = []
    NormImageThubnails = []
    ShownImagesThumbnailsSizes = []
    ShownImageLocation = 4

    def NewSetOptions():
        NewSetEntry.place_forget()
        NewSetLabel.place_forget()
        NewSetButton.place_forget()
        if NormKeep_Var.get() == 1:
            NewSetLabel.place(anchor="center", relx=0.15, rely=0.85)
            NewSetEntry.place(anchor="center", relx=0.13, rely=0.9)
            NewSetButton.place(anchor="center", relx=0.23, rely=0.9)

    def GetImage():
        Step1Entry.delete(0, 10000)
        TargetImage.clear()

        Target_Image = filedialog.askopenfilename(initialdir=os.getcwd(), title="Please select your top-level folder.",
                                                  filetypes=[("Image", "*.jpg"), ("Image", "*.png"),
                                                             ("Image", "*.jfif")])
        Step1Entry.insert(0, str(Target_Image))
        Resized_Target = Image.open(Step1Entry.get())
        print(type(Resized_Target))
        CanvasSize = 300, 300
        Resized_Target.thumbnail(CanvasSize, Image.ANTIALIAS)
        img = ImageTk.PhotoImage(Resized_Target)
        TargetImage.append(img)

        for I in TargetImage:
            Norm_Image_Canvas.create_image(150, 150, image=I, anchor="center")

        if len(ShownImagesThumbnails) != 0:
            LoadPreview()

    def EnterGetImage(e):
        Step1Text.config(fg="black", text="Please select the image you want to normalise to:")
        TargetImage.clear()
        print(Step1Entry.get())
        try:
            Resized_Target = Image.open(Step1Entry.get())
            print(type(Resized_Target))
            CanvasSize = 300, 300
            Resized_Target.thumbnail(CanvasSize, Image.ANTIALIAS)
            img = ImageTk.PhotoImage(Resized_Target)
            TargetImage.append(img)

            for I in TargetImage:
                Norm_Image_Canvas.create_image(150, 150, image=I, anchor="center")

            if len(ShownImagesThumbnails) != 0:
                LoadPreview()
        except:
            Step1Text.config(fg="red", text="Invalid Image Path, please try again")

    def LoadImages():
        nonlocal ShownImageLocation
        Step2Entry.delete(0, 10000)
        ImagePaths.clear()
        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        filename2 = filedialog.askdirectory(initialdir=os.getcwd(), title="Please select your top-level folder.")
        Step2Entry.insert(0, filename2)
        for r, d, f in os.walk(Step2Entry.get()):
            for file in f:
                if '.jpg' in file or '.jfif' in file or ".png" in file:
                    ImagePaths.append(os.path.join(r, file))

        ActiveImages = ImagePaths[:ShownImageLocation]
        for A in ActiveImages:
            ShownImagesPaths.append(A)

        # Make Thumbnail Images
        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            CanvasSize = 220, 220
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)

        CanvasXLoc = 0

        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 1000 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 3)
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            OrigCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

        if len(TargetImage) != 0:
            LoadPreview()

    def EnterLoadImages(e):
        Step2Text.config(fg="black", text="Please select images for normalisation:")
        ImagePaths.clear()
        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()

        for r, d, f in os.walk(Step2Entry.get()):
            for file in f:
                if '.jpg' in file or '.jfif' in file or ".png" in file:
                    ImagePaths.append(os.path.join(r, file))

        if len(ImagePaths) == 0:

            Step2Text.config(fg="red", text="Invalid Folder Path, please try again")

        else:
            ActiveImages = ImagePaths[:ShownImageLocation]
            for A in ActiveImages:
                ShownImagesPaths.append(A)

            # Make Thumbnail Images
            for Img in ShownImagesPaths:
                Im = Image.open(Img)
                CanvasSize = 220, 220
                Im.thumbnail(CanvasSize, Image.ANTIALIAS)
                ThumbnailSize = Im.size
                ShownImagesThumbnailsSizes.append(ThumbnailSize)
                PreviewImage = ImageTk.PhotoImage(Im)
                ShownImagesThumbnails.append(PreviewImage)

            CanvasXLoc = 0

            TotalWidth = 0
            for I in ShownImagesThumbnailsSizes:
                width = I[0]
                TotalWidth += width

            DeadSpace = 1000 - TotalWidth
            DeadspaceFraction = round(DeadSpace / 3)
            for I in range(len(ShownImagesThumbnails)):
                Nigel = ShownImagesThumbnails[I]
                width, height = ShownImagesThumbnailsSizes[I]
                OrigCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
                CanvasXLoc += (width + DeadspaceFraction)

            if len(TargetImage) != 0:
                LoadPreview()

    def LoadPreview():
        NormCanvas.delete("all")
        NormImageThubnails.clear()

        for I in ShownImagesPaths:
            if TypeDropdown.get() == "Histogram Equalisation":
                Canvas_Size = 220, 220
                Orig_Image = io.imread(I)
                Target_Image = Image.open(str(Step1Entry.get()))
                Target_Image = np.array(Target_Image)
                if Target_Image.shape[2] != 3:
                    print("Incorrect number of channels - rectifying")
                    print(Target_Image)
                    Target_Image = Target_Image[:, :, 0]
                    Target_Image = cv2.cvtColor(Target_Image, cv2.COLOR_GRAY2RGB)
                    Target_Image = Image.fromarray(Target_Image)
                else:
                    Target_Image = Image.fromarray(Target_Image)

                try:
                    Target_Image.thumbnail(Canvas_Size, Image.ANTIALIAS)
                    Target_Image = np.array(Target_Image)
                    Norm_Image = match_histograms(Orig_Image, Target_Image, multichannel=True)
                    Norm_Image = Image.fromarray(Norm_Image)
                    Norm_Image.thumbnail(Canvas_Size, Image.ANTIALIAS)
                    Norm_Image = ImageTk.PhotoImage(Norm_Image)
                    NormImageThubnails.append(Norm_Image)
                except:

                    Target_Image.thumbnail(Canvas_Size, Image.ANTIALIAS)
                    Target_Image = np.array(Target_Image)
                    print(Target_Image)
                    print(Target_Image.shape)
                    Target_Image = cv2.cvtColor(Target_Image, cv2.COLOR_GRAY2RGB)
                    print(Target_Image)
                    Norm_Image = match_histograms(Orig_Image, Target_Image, multichannel=True)
                    Norm_Image = Image.fromarray(Norm_Image)
                    Norm_Image.thumbnail(Canvas_Size, Image.ANTIALIAS)
                    Norm_Image = ImageTk.PhotoImage(Norm_Image)
                    NormImageThubnails.append(Norm_Image)


            elif TypeDropdown.get() == "Reinhard Method":
                Canvas_Size = 220, 220
                image = cv2.imread(I)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                original = cv2.imread(Step1Entry.get())
                original = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

                def getavgstd(image):
                    avg = []
                    std = []
                    image_avg_l = np.mean(image[:, :, 0])
                    image_std_l = np.std(image[:, :, 0])
                    image_avg_a = np.mean(image[:, :, 1])
                    image_std_a = np.std(image[:, :, 1])
                    image_avg_b = np.mean(image[:, :, 2])
                    image_std_b = np.std(image[:, :, 2])
                    avg.append(image_avg_l)
                    avg.append(image_avg_a)
                    avg.append(image_avg_b)
                    std.append(image_std_l)
                    std.append(image_std_a)
                    std.append(image_std_b)
                    return (avg, std)

                image_avg, image_std = getavgstd(image)
                original_avg, original_std = getavgstd(original)

                height, width, channel = image.shape
                for i in range(0, height):
                    for j in range(0, width):
                        for k in range(0, channel):
                            t = image[i, j, k]
                            t = (t - image_avg[k]) * (original_std[k] / image_std[k]) + original_avg[k]
                            t = 0 if t < 0 else t
                            t = 255 if t > 255 else t
                            image[i, j, k] = t
                image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
                image = Image.fromarray(image)
                image.thumbnail(Canvas_Size, Image.ANTIALIAS)
                Norm_Image = ImageTk.PhotoImage(image)
                NormImageThubnails.append(Norm_Image)

        CanvasXLoc = 0
        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 1000 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 3)
        print(ShownImagesThumbnails)
        print(NormImageThubnails)
        for I in range(len(ShownImagesThumbnails)):
            Nigel = NormImageThubnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            NormCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

    def NextSet():
        nonlocal ShownImageLocation

        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        NormImageThubnails.clear()
        ShownImageLocation += 4

        if ShownImageLocation >= len(ImagePaths):
            ShownImageLocation = len(ImagePaths)

        First_Image = ShownImageLocation - 4

        if First_Image <= 0:
            First_Image = 0

        for x in ImagePaths[First_Image:ShownImageLocation]:
            ShownImagesPaths.append(x)

        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            print(type(Im))
            CanvasSize = 220, 220
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            print(ThumbnailSize)

            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)

        CanvasXLoc = 0

        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 1000 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 3)
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            print(width, height)
            OrigCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

        try:
            LoadPreview()
        except:
            pass

    def PrevSet():
        nonlocal ShownImageLocation

        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        NormImageThubnails.clear()
        ShownImageLocation -= 4

        if ShownImageLocation <= 4:
            ShownImageLocation = 4

        First_Image = ShownImageLocation - 4

        if First_Image <= 0:
            First_Image = 0

        for x in ImagePaths[First_Image:ShownImageLocation]:
            ShownImagesPaths.append(x)

        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            print(type(Im))
            CanvasSize = 220, 220
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            print(ThumbnailSize)

            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)

        CanvasXLoc = 0

        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 1000 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 3)
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            print(width, height)
            OrigCanvas.create_image(CanvasXLoc, 150, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

        try:
            LoadPreview()
        except:
            pass

    def GetSaveLocation():
        WarnLabel.config(text="")
        Filecount = 0
        NewSetEntry.delete(0, 10000)
        filename2 = filedialog.askdirectory(initialdir=os.getcwd(), title="Please select your top-level folder.")
        NewSetEntry.insert(0, filename2)

        for r, d, f in os.walk(NewSetEntry.get()):
            for file in f:
                Filecount += 1

        if Filecount > 0:
            WarnLabel.config(text="WARNING: Folder contains other files.\n"
                                  "It is recommended that the output folder is empty.")

    def PreChecks():
        Errors = 0
        Step1Text.config(fg="black", text="Step 1: Please select the image you want to normalise to:")
        Step2Text.config(fg="black", text="Step 2: Please select images for normalisation:")
        NewSetLabel.config(fg="black", text="Select save folder for new images")

        if len(Step1Entry.get()) == 0:
            Step1Text.config(fg="red")
            Errors += 1
        elif not os.path.isfile(str(Step1Entry.get())):
            Step1Text.config(fg="red", text="Not a valid Image")
            Errors += 1

        if len(Step2Entry.get()) == 0:
            Step2Text.config(fg="red")
            Errors += 1
        elif not os.path.isdir(str(Step2Entry.get())):
            Step2Text.config(fg="red", text="Not a valid folder")
            Errors += 1
        elif len(ImagePaths) == 0:
            Step2Text.config(fg="red", text="No Images in selected folder")
            Errors += 1

        if NormKeep_Var.get() == 1:
            if len(NewSetEntry.get()) == 0:
                NewSetLabel.config(fg="red")
                Errors += 1
            elif not os.path.isdir(str(NewSetEntry.get())):
                NewSetLabel.config(fg="red", text="Not a valid folder")
                Errors += 1
        return Errors

    def NormHist(ImagePath, Target):
        global Images_Processed
        # print(ImagePath)
        Orig_Image = io.imread(ImagePath)

        Norm_Image = match_histograms(Orig_Image, Target, multichannel=True)

        if NormKeep_Var.get() == 1:
            Chop = ImagePath.replace(Step2Entry.get(), "")
            NewLoc = str(NewSetEntry.get()) + Chop
            print(NewLoc)
            io.imsave(NewLoc, Norm_Image)
            Images_Processed += 1

        else:
            io.imsave(ImagePath, Norm_Image)
            Images_Processed += 1

        print(Images_Processed)

    def ReinHist(ImagePath):
        global Images_Processed

        image = cv2.imread(ImagePath)
        # print(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        original = cv2.imread(Step1Entry.get())
        # print(original)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

        def getavgstd(image):
            avg = []
            std = []
            image_avg_l = np.mean(image[:, :, 0])
            image_std_l = np.std(image[:, :, 0])
            image_avg_a = np.mean(image[:, :, 1])
            image_std_a = np.std(image[:, :, 1])
            image_avg_b = np.mean(image[:, :, 2])
            image_std_b = np.std(image[:, :, 2])
            avg.append(image_avg_l)
            avg.append(image_avg_a)
            avg.append(image_avg_b)
            std.append(image_std_l)
            std.append(image_std_a)
            std.append(image_std_b)
            return (avg, std)

        getavgstdjit = jit(nopython=True)(getavgstd)

        image_avg, image_std = getavgstdjit(image)
        # print(image_avg)
        original_avg, original_std = getavgstdjit(original)
        # print(original_avg)

        height, width, channel = image.shape
        for i in range(0, height):
            for j in range(0, width):
                for k in range(0, channel):
                    t = image[i, j, k]
                    t = (t - image_avg[k]) * (original_std[k] / image_std[k]) + original_avg[k]
                    t = 0 if t < 0 else t
                    t = 255 if t > 255 else t
                    image[i, j, k] = t
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        print(type(image))
        print(image.shape)

        norm_image = Image.fromarray(image)
        # print(norm_image)

        if NormKeep_Var.get() == 1:
            Images_Processed += 1
            print(Images_Processed)
            Chop = ImagePath.replace(Step2Entry.get(), "")
            NewLoc = str(NewSetEntry.get()) + Chop
            # print(NewLoc)
            # print(NewSetEntry.get())
            norm_image.save(NewLoc)


        else:
            Images_Processed += 1
            print(Images_Processed)
            print(ImagePath)
            norm_image.save(ImagePath)
            # Processed_Images += 1

    def Norm_Thread_Threading():
        Errors = PreChecks()

        if Errors == 0:
            if NormKeep_Var.get() == 1:
                RecreateFolderStructure()
            t = threading.Thread(target=Norm_Threading)
            t.start()
            ProgressPopup(len(ImagePaths))

    def Norm_Threading():
        TargetImage = io.imread(str(Step1Entry.get()))

        if TypeDropdown.get() == "Histogram Equalisation":

            if len(ImagePaths) > 0:
                pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
                for I in range(len(ImagePaths)):
                    # print(ImagePaths[I])
                    pool.submit(NormHist, ImagePaths[I], TargetImage)
                pool.shutdown(wait=True)
        elif TypeDropdown.get() == "Reinhard Method":
            print("Running Reinhard Method")
            if len(ImagePaths) > 0:
                pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
                for I in range(len(ImagePaths)):
                    # print(ImagePaths[I])
                    pool.submit(ReinHist, ImagePaths[I])
                pool.shutdown(wait=True)

    def RecreateFolderStructure():
        OriginalTopFolder = Step2Entry.get()
        NewSetTopFolder = NewSetEntry.get()

        for In in ImagePaths:

            OriginalImageLocation = os.path.split(In)[0]
            OriginalImageShortLocation = OriginalImageLocation.replace(OriginalTopFolder, "")
            # os.path.join kept fucking up here, no idea why.
            NewFolder = str(NewSetTopFolder) + str(OriginalImageShortLocation)  # + "\\" + str(OriginalImageName)

            try:
                os.makedirs(NewFolder)
            except Exception as e:
                pass

    Step1Text = tk.Label(MainFrame, fg="black", text="Please select the image you want to normalise to:", bg="gray83")
    Step1Text.place(anchor="center", relx=0.15, rely=0.05)

    Step1Entry = tk.Entry(MainFrame, width=25, font=Large_Font)
    Step1Entry.place(anchor="center", relx=0.12, rely=0.1)
    Step1Entry.bind('<Return>',func=EnterGetImage)

    ButtonForFiles = tk.Button(MainFrame, text="Browse...", command=GetImage)
    ButtonForFiles.place(anchor="center", relx=0.24, rely=0.1)

    Step2Text = tk.Label(MainFrame, fg="black", text="Please select images for normalisation:", bg="gray83")
    Step2Text.place(anchor="center", relx=0.15, rely=0.17)


    Step2Entry = tk.Entry(MainFrame, width=25, font=Large_Font)
    Step2Entry.place(anchor="center", relx=0.12, rely=0.22)
    Step2Entry.bind('<Return>', func=EnterLoadImages)

    ButtonForFiles2 = tk.Button(MainFrame, text="Browse...", command=LoadImages)
    ButtonForFiles2.place(anchor="center", relx=0.24, rely=0.22)

    Norm_Image_Label = tk.Label(MainFrame, text="Target Image:", bg="gray83", font=Medium_Font)
    Norm_Image_Label.place(relx=0.15, rely=0.27, anchor="center")

    Norm_Image_Canvas = tk.Canvas(MainFrame, width=300, height=300)
    Norm_Image_Canvas.place(anchor="center", relx=0.15, rely=0.5)

    OrigLabel = tk.Label(MainFrame, text="Original Images", font=Large_Font, bg="gray83")
    OrigLabel.place(anchor="center", relx=0.65, rely=0.025)
    OrigCanvas = tk.Canvas(MainFrame, width=1000, height=300)
    OrigCanvas.place(anchor="center", relx=0.65, rely=0.22)

    NormLabel = tk.Label(MainFrame, text="Normalised Images", font=Large_Font, bg="gray83")
    NormLabel.place(anchor="center", relx=0.65, rely=0.41)
    NormCanvas = tk.Canvas(MainFrame, width=1000, height=300)
    NormCanvas.place(anchor="center", relx=0.65, rely=0.6)

    NormKeep_Var = tk.IntVar(value=1)
    KeepImageCheck = tk.Checkbutton(MainFrame, variable=NormKeep_Var, offvalue=0, onvalue=1,
                                    text="Keep Original Images?", bg="gray83", command=NewSetOptions)
    KeepImageCheck.place(anchor="center", relx=0.15, rely=0.8)

    NewSetLabel = tk.Label(MainFrame, text="Select save folder for new images", bg="gray83")
    NewSetEntry = tk.Entry(MainFrame,  width=25, font=Large_Font)
    NewSetButton = tk.Button(MainFrame, text="Browse...", command=GetSaveLocation)

    NewSetLabel.place(anchor="center", relx=0.15, rely=0.85)
    NewSetEntry.place(anchor="center", relx=0.12, rely=0.9)
    NewSetButton.place(anchor="center", relx=0.24, rely=0.9)

    TypeLabel = tk.Label(MainFrame, text="Select Normalisation Type", bg="gray83")
    TypeLabel.place(anchor="center", relx=0.15, rely=0.7)

    OptionList = ["Histogram Equalisation", "Reinhard Method"]
    TypeDropdown = ttk.Combobox(MainFrame, values=OptionList, width=35, state="readonly")
    TypeDropdown.set(OptionList[0])
    TypeDropdown.place(anchor="center", relx=0.15, rely=0.75)
    TypeDropdown.bind("<<ComboboxSelected>>", lambda e: LoadPreview())

    WarnLabel = tk.Label(MainFrame, text="", fg="red", bg="gray83")
    WarnLabel.place(anchor="center", relx=0.15, rely=0.94)

    Prev_Img_Button = tk.Button(MainFrame, text="Previous Set", width=11, command=PrevSet)
    Prev_Img_Button.place(anchor="center", relx=0.6, rely=0.85)

    Next_Img_Button = tk.Button(MainFrame, text="Next Set", width=11, command=NextSet)
    Next_Img_Button.place(anchor="center", relx=0.675, rely=0.85)

    Begin_Label = tk.Label(MainFrame, fg="black", text="Press to begin -->", bg="gray83")
    Begin_Label.place(anchor="center", relx=0.78, rely=0.85)

    BeginButton = tk.Button(MainFrame, text="Begin Normalisation", height=1, width=15,
                                command=Norm_Thread_Threading)
    BeginButton.place(anchor="center", relx=0.87, rely=0.85)

def AugmentationModule():
    global Active_Module
    Active_Module = "Aug"
    # Variables For adding options
    ShownImageLocation = 4
    ImageCount = 0
    Processed_Images = 0

    ActiveOption = 0


    InnerFramesList = []
    InnerFramesIDs = []
    OptionWidgetListID = []
    LabelList = []
    OPButtList = []
    RemButtList = []
    TypeDropDownList = []
    DescriptionList = []

    ActiveWidgetList = []
    SavedWidgetList = []
    SavedWidgetValueList = []

    OptionHistory = []  # History for the Option seection for each
    SubOptionHistory = []  # History of SubOption Selection for each

    SavedOptionsList = []  # Values for each of the augmentations
    SavedOptionCommand = []  # Resulting Augmentation Commands

    ImagePaths = []
    ImagesForAug = []
    ShownImagesPaths = []
    ShownImagesThumbnails = []
    ShownImagesThumbnailsSizes = []
    PreviewImageThumbnails = []

    # Functions for Adding and Removing Options

    AugOptions = 0
    ActiveAugmentations = 1
    Yplace = 120
    BottomEdge = 120

    def AddOption():
        nonlocal AugOptions
        nonlocal Yplace
        nonlocal BottomEdge
        nonlocal ActiveAugmentations
        nonlocal OptionHistory
        nonlocal SubOptionHistory
        nonlocal InnerFramesList


        ActiveVar = tk.IntVar()
        # print(f"Type = {(type(ActiveVar))}")
        AugOptions += 1

        ActiveAugmentations += 1
        InnerFramesIDs.append(ActiveAugmentations)
        OptionHistory.append(" ")
        SubOptionHistory.append(" ")
        SavedOptionsList.append(" ")
        SavedOptionCommand.append([])
        SavedWidgetValueList.append([])

        AugFrameInner = tk.Frame(AugCanvas, height=100, width=380, relief="raised", borderwidth=1, bg="gray88")
        AugFrameInner.ID = ActiveAugmentations


        InnerFramesList.append(AugFrameInner)



        AugLabel = tk.Label(AugFrameInner, text=f"Augmentation {AugOptions+1}", name=f"label{AugOptions}",
                            bg="gray88")
        AugLabel.ID = f"Aug Label {ActiveAugmentations}"
        AugLabel.place(relx=0.5, rely=0.2, anchor="center")
        LabelList.append(AugLabel)



        DescriptionLabel = tk.Label(AugFrameInner, text="", name=f"desclabel{AugOptions}", bg="gray88")
        DescriptionLabel.place(relx=0.5, rely=0.8, anchor="center")
        DescriptionLabel.ID = f"Desc Label {ActiveAugmentations}"
        DescriptionList.append(DescriptionLabel)

        OptionButton = tk.Button(AugFrameInner, text="Open", name=f"options{AugOptions}",
                                 command=lambda: OpenOption(str(OptionButton)), bg="gray88")
        OptionButton.place(relx=0.85, rely=0.3, anchor="center")
        DescriptionLabel.ID = f"Open Button {ActiveAugmentations}"
        OPButtList.append(str(OptionButton))


        OptionList = ["Colour Options","Size Transformations", "Brightness and Contrast", "Sharpen and Emboss", "Noise", "Dropout", "Blur",
                      "Geometric Transformations", "Edge Detection", "Colour Segmentation", "Artistic Options"]
        TypeDropdown = ttk.Combobox(AugFrameInner, values=OptionList, width=35, state="readonly",
                                    name=f"dropdown{AugOptions}")
        TypeDropdown.unbind_class("TCombobox", "<MouseWheel>")

        TypeDropdown.set("Select Augmentation Type")
        TypeDropdown.place(relx=0.4, rely=0.5, anchor="center")
        DescriptionLabel.ID = f"Dropdown {ActiveAugmentations}"
        TypeDropDownList.append(TypeDropdown)
        TypeDropdown.bind("<<ComboboxSelected>>", lambda e: OpenOption(str(OptionButton)))

        RemoveButton = tk.Button(AugFrameInner, text="Remove", name=f"remove{AugOptions}", bg="gray88", command=lambda: RemoveOption(RemoveButton))
        RemoveButton.ID = f"Remove Button {ActiveAugmentations}"
        RemoveButton.place(relx=0.85, rely=0.7, anchor="center")
        RemButtList.append(str(RemoveButton))


        ID = AugCanvas.create_window(210, Yplace, window=AugFrameInner, anchor="n", tags=f"{ActiveAugmentations}")
        #ID.uniqueID = f"Frame {ActiveAugmentations}"


        Yplace += 120
        BottomEdge += 120
        if BottomEdge > 300:
            AugCanvas.config(scrollregion=(0, 0, 500, BottomEdge))
        OptionWidgetListID.append(ID)

        print("Add- ActiveAugmentations: ", ActiveAugmentations)
        print("Add-Option History: ", OptionHistory)
        print("Add-SubOptionHistory: ", SubOptionHistory)
        print("Add-Saved Options List: ", SavedWidgetValueList)
        print("Add-Saved Options Command:", SavedOptionCommand)
        #print("Canvas Widgit IDs ", OptionWidgetListID)
        #print("Frame IDs: ", InnerFramesIDs)
        #print(Yplace, BottomEdge)

    def RemoveOption(ButtonNumber):
        nonlocal AugOptions
        nonlocal Yplace
        nonlocal BottomEdge
        nonlocal ActiveAugmentations
        nonlocal ActiveOption
        nonlocal InnerFramesList

        for Child in OptionsFrame.winfo_children():
            Child.place_forget()

        #print("_____________________________")
        # Match the button to the location of the button in OPButtList
        print("Button Clicked : " , str(ButtonNumber))
        print("Active Augmentations Before Removal: ", ActiveAugmentations )
        #print(RemButtList)

        IndexForRemoval = RemButtList.index(str(ButtonNumber))



        # Note, FrameToRemoveID is the name of the frame and not the index
        #FrameToRemoveID = int(ButtonNumber.ID[-1])
        #print(FrameToRemoveID)

        if ActiveAugmentations > 1:
            #print(f"Frame Number {FrameToRemoveID} removed")
            #print("OptionWidgetListID: ", OptionWidgetListID)
            AugCanvas.delete(OptionWidgetListID[IndexForRemoval])
            #print(OptionWidgetListID)
            del LabelList[IndexForRemoval]
            del OPButtList[IndexForRemoval]
            del InnerFramesList[IndexForRemoval]
            del InnerFramesIDs[IndexForRemoval]
            del DescriptionList[IndexForRemoval]

            del SavedOptionsList[IndexForRemoval]
            del SavedOptionCommand[IndexForRemoval]
            del SavedWidgetValueList[IndexForRemoval]

            del OptionHistory[IndexForRemoval]
            del TypeDropDownList[IndexForRemoval]
            del SubOptionHistory[IndexForRemoval]

            del RemButtList[IndexForRemoval]

            #print(OptionWidgetListID)

            if BottomEdge > 300:
                AugCanvas.config(scrollregion=(0, 0, 500, BottomEdge))
            else:
                AugCanvas.config(scrollregion=(0, 0, 500, 300))

            for I in range(len(InnerFramesIDs)):
                if InnerFramesIDs[I] > IndexForRemoval:
                    InnerFramesIDs[I] -= 1

            print("Index ", IndexForRemoval)
            print("Canvas Widgit IDs ", OptionWidgetListID)

            for x in OptionWidgetListID:

                if OptionWidgetListID[IndexForRemoval] < x:
                    #print(x)
                    #print(IndexForRemoval)
                    AugCanvas.move(x, 0, -120)
            del OptionWidgetListID[IndexForRemoval]
            ### Update Aug Labels ##
            for L in range(len(LabelList)):
                LabelList[L].config(text=f"Augmentation {InnerFramesIDs[L]}")

            BottomEdge -= 120
            Yplace -= 120
            if BottomEdge > 300:
                AugCanvas.config(scrollregion=(0, 0, 500, BottomEdge))
            else:
                AugCanvas.config(scrollregion=(0, 0, 500, 300))

            ActiveAugmentations -= 1
            AugOptions -= 1
            print("Active Augmentations After Removal: ", ActiveAugmentations)
        else:
            print("Must have one augmentation option present")

        try:
            PreviewAugmentations()

        except:
            pass



       # print(InnerFramesIDs)

        #print("__________________________________")
        #print(InnerFramesIDs)
        #print(len(InnerFramesIDs))
        #print(len(LabelList))
        #print("__________________________________")
        #print(OptionWidgetListID)




        #for I in range(len(OptionWidgetListID)):
           # print(OptionWidgetListID[InnerFramesIDs[I]-1])
           # AugCanvas.itemcget(OptionWidgetListID[InnerFramesIDs[I]-1], option="text")



        #print("Remove-ActiveAugmentations: ", ActiveAugmentations)
        print("Remove-Option History: ", OptionHistory)
        print("Remove-SubOptionHistory: ", SubOptionHistory)
        print("Remove-Saved Options List: ", SavedWidgetValueList)
        print("Remove-Saved Options Command:", SavedOptionCommand)


        #print(Yplace, BottomEdge)

    def OpenOption(ButtonNumber):
        nonlocal ActiveOption
        print(OPButtList)
        IndexToOpen = OPButtList.index(ButtonNumber)
        print(IndexToOpen)
        SubDescriptionLabel.config(text="")

        for c in OptionsFrame.winfo_children():
            c.place_forget()

        for x in ActiveWidgetList:
            x.place_forget()

        DescLabel.config(text="")
        Num = int(ButtonNumber[-1])  # Identify which options button was pressed
        print(Num)

        ActiveOption = IndexToOpen

        for I in range(len(InnerFramesList)):
            InnerFramesList[I].config(relief="raised", bg="gray88")
            LabelList[I].config(bg="gray88")
            DescriptionList[I].config(bg="gray88")

        InnerFramesList[IndexToOpen].config(relief="sunken", bg="gray64")
        LabelList[IndexToOpen].config(bg="gray64")
        DescriptionList[IndexToOpen].config(bg="gray64")
        LabelList[IndexToOpen].config(fg="black", text=f"Augmnetation {Num + 1}")

        try:
            del OptionHistory[IndexToOpen]
        except:
            pass
        OptionHistory.insert(IndexToOpen, TypeDropDownList[IndexToOpen].get())

        Augtype = TypeDropDownList[IndexToOpen].get()

        try:
            LastSub = SubOptionHistory[IndexToOpen]
        except:
            print("No History for this detected")
            LastSub = ""

        for L in range(len(LabelList)):
            LabelList[L].config(fg="black", text=f"Augmentation {L+1}")

        if Augtype == "Select Augmentation Type":
            LabelList[IndexToOpen].config(fg="red", text="Please select annotation option")
        else:
            # print(f"Showing Options for {Augtype}")
            pass

        if Augtype == "Colour Options":
            DisplayColourOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Size Transformations":
            DisplaySizeOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Brightness and Contrast":
            DisplayBriConOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Sharpen and Emboss":
            DisplaySharpEmbOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Noise":
            DisplayNoiseOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Dropout":
            DisplayDropoutOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Blur":
            DisplayBlurOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Geometric Transformations":
            print("Correct")
            DisplayGeoOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Edge Detection":
            DisplayEdgeOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Colour Segmentation":
            DisplayColourSegOptions(IndexToOpen, OptionsFrame, LastSub)

        if Augtype == "Artistic Options":
            DisplayArtSegOptions(IndexToOpen, OptionsFrame, LastSub)

        ##################################### Option Displays ############################################

    def PreviewAugmentations():
        PreviewImageThumbnailSizes = []


        PreviewImageThumbnails.clear()
        try:
            PrevCanvas.delete("all")
        except:
            pass

        Flat_Aug = []
        print("In Preview Augmentations")
        for x in SavedOptionCommand:
            print(x)

        for x in SavedOptionCommand:
            if type(x) is list:
                for item in x:
                    Flat_Aug.append(item)
            else:
                Flat_Aug.append(x)

        if len(Flat_Aug) > 1:
            Augmentation = iaa.Sequential(Flat_Aug)
        else:
            Augmentation = Flat_Aug[0]

        for x in ShownImagesPaths:
            Image_in = imageio.imread(x)
            Image_Out = Augmentation.augment_image(Image_in)
            Image_Out = Image.fromarray(Image_Out)
            CanvasSize = 200, 200
            Image_Out.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Image_Out.size
            PreviewImageThumbnailSizes.append(ThumbnailSize)
            PreviewImage = ImageTk.PhotoImage(Image_Out)
            PreviewImageThumbnails.append(PreviewImage)



        TotalWidth = 0
        for I in PreviewImageThumbnailSizes:
            width = I[0]
            TotalWidth += width
        print("TotalWidth: ", TotalWidth)

        DeadSpace = 900 - TotalWidth
        print("DeadSpace: ", DeadSpace)
        DeadspaceFraction = round(DeadSpace / 5)
        CanvasXLoc = DeadspaceFraction
        print("DeadspaceFraction: ", DeadspaceFraction)
        for I in range(len(PreviewImageThumbnails)):
            Nigel = PreviewImageThumbnails[I]
            width, height = PreviewImageThumbnailSizes[I]
            PrevCanvas.create_image(CanvasXLoc, 125, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)
            print("CanvasXLoc:", CanvasXLoc)

    def DisplayColourOptions(Number, Frame, LastSub):
        No = Number

        ColourOptions = ["Invert Colours", "RGB Levels", "Saturation", "Greyscale",
                         "Temprature", "K-means Colour Quantization"]
        SubTypeDropdown = ttk.Combobox(Frame, values=ColourOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())
        print("Should be Setting")

        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")
        SubTypeDropdown.set("Select Colour adjustment option")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Invert Colours":
                AugList = []
                if RedInvVar.get() == 1:
                    AugList.append(iaa.WithChannels(0, iaa.Invert(1)))
                if GreenInvVar.get() == 1:
                    AugList.append(iaa.WithChannels(1, iaa.Invert(1)))
                if BlueInvVar.get() == 1:
                    AugList.append(iaa.WithChannels(2, iaa.Invert(1)))

                ImportantWidgets.extend((RedInvert, GreenInvert, BlueInvert))
                ImportantWidgetValues.extend((RedInvVar.get(), GreenInvVar.get(), BlueInvVar.get()))

                try:
                    # del SavedOptionCommand[No]
                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetList.insert(No, ImportantWidgets)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

                print(SavedWidgetList)
                print("SavedWidgetValueList: ", SavedWidgetValueList)

            if SubTypeDropdown.get() == "RGB Levels":
                AugList = []

                if RedLevelVar.get() == 1:
                    AugList.append(
                        iaa.WithChannels(0, iaa.Add((int(RedEntryMin.get()), int(RedEntryMax.get())), per_channel=1)))
                    ImportantWidgetValues.append(RedLevelVar.get())
                else:
                    ImportantWidgetValues.append(RedLevelVar.get())

                if GreenLevelVar.get() == 1:
                    AugList.append(iaa.WithChannels(1, iaa.Add((int(GreenEntryMin.get()), int(GreenEntryMax.get())),
                                                               per_channel=1)))
                    ImportantWidgetValues.append(GreenLevelVar.get())

                else:
                    ImportantWidgetValues.append(GreenLevelVar.get())

                if BlueLevelVar.get() == 1:
                    AugList.append(iaa.WithChannels(2, iaa.Add((int(BlueEntryMin.get()), int(BlueEntryMax.get())),
                                                               per_channel=1)))
                    ImportantWidgetValues.append(BlueLevelVar.get())
                else:
                    ImportantWidgetValues.append(BlueLevelVar.get())

                if RedLevelVar.get() == 1:
                    ImportantWidgets.extend((RedEntryMin, RedEntryMax))
                    ImportantWidgetValues.extend((RedEntryMin.get(), RedEntryMax.get()))
                else:
                    ImportantWidgets.extend((RedEntryMin, RedEntryMax))
                    ImportantWidgetValues.extend((0, 0))

                if GreenLevelVar.get() == 1:
                    ImportantWidgets.extend((GreenEntryMin, GreenEntryMax))
                    ImportantWidgetValues.extend((GreenEntryMin.get(), GreenEntryMax.get()))
                else:
                    ImportantWidgets.extend((GreenEntryMin, GreenEntryMax))
                    ImportantWidgetValues.extend((0, 0))

                if BlueLevelVar.get() == 1:
                    ImportantWidgets.extend((BlueEntryMin, BlueEntryMax))
                    ImportantWidgetValues.extend((BlueEntryMin.get(), BlueEntryMax.get()))
                else:
                    ImportantWidgets.extend((BlueEntryMin, BlueEntryMax))
                    ImportantWidgetValues.extend((0, 0))

                try:
                    # del SavedOptionCommand[No]
                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetList.insert(No, ImportantWidgets)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Saturation":
                AugList = []

                AugList.append(iaa.AddToSaturation((int(SatEntryMin.get()), int(SatEntryMax.get()))))
                ImportantWidgetValues.extend((int(SatEntryMin.get()), int(SatEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Greyscale":
                AugList = []

                AugList.append(iaa.Grayscale((float(GrayEntryMin.get()), float(GrayEntryMax.get()))))
                ImportantWidgetValues.extend((float(GrayEntryMin.get()), float(GrayEntryMax.get())))

                try:
                    # del SavedOptionCommand[No]
                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Temprature":
                AugList = []

                AugList.append(iaa.ChangeColorTemperature((int(TempEntryMin.get()), int(TempEntryMax.get()))))
                ImportantWidgetValues.extend((int(TempEntryMin.get()), int(TempEntryMax.get())))

                try:
                    # del SavedOptionCommand[No]
                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "K-means Colour Quantization":
                AugList = []

                AugList.append(iaa.KMeansColorQuantization((int(KMeanEntryMin.get()), int(KMeanEntryMax.get()))))
                ImportantWidgetValues.extend((int(KMeanEntryMin.get()), int(KMeanEntryMax.get())))

                try:
                    # del SavedOptionCommand[No]
                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)
            try:
                PreviewAugmentations()
            except:
                pass

        DefaultCol = "gray86"

        OptionsText.place_forget()

        ################################################## Invert Colours Widgets ####################################

        RedInvVar = tk.IntVar()
        # CheckWidgetValues.append(RedInvVar)
        GreenInvVar = tk.IntVar()
        # CheckWidgetValues.append(GreenInvVar)
        BlueInvVar = tk.IntVar()
        # CheckWidgetValues.append(BlueInvVar)

        RedInvert = tk.Checkbutton(Frame, variable=RedInvVar, onvalue=1, offvalue=0, bg="gray86")
        GreenInvert = tk.Checkbutton(Frame, variable=GreenInvVar, onvalue=1, offvalue=0, bg="gray86")
        BlueInvert = tk.Checkbutton(Frame, variable=BlueInvVar, onvalue=1, offvalue=0, bg="gray86")

        ########################################## RGB Levels Widgets #################################################

        RedLevelVar = tk.IntVar()
        # CheckWidgetValues.append(RedInvVar)
        GreenLevelVar = tk.IntVar()
        # CheckWidgetValues.append(GreenInvVar)
        BlueLevelVar = tk.IntVar()
        # CheckWidgetValues.append(BlueInvVar)

        RedLevelCheck = tk.Checkbutton(Frame, variable=RedLevelVar, onvalue=1, offvalue=0, bg="gray86")
        GreenLevelCheck = tk.Checkbutton(Frame, variable=GreenLevelVar, onvalue=1, offvalue=0, bg="gray86")
        BlueLevelCheck = tk.Checkbutton(Frame, variable=BlueLevelVar, onvalue=1, offvalue=0, bg="gray86")

        RedEntryMin = tk.Entry(Frame, width=6)
        RedEntryMax = tk.Entry(Frame, width=6)
        RedLabelMin = tk.Label(Frame, text="", bg="gray86")
        RedLabelMax = tk.Label(Frame, text="", bg="gray86")

        GreenEntryMin = tk.Entry(Frame, width=6)
        GreenEntryMax = tk.Entry(Frame, width=6)
        GreenLabelMin = tk.Label(Frame, text="", bg="gray86")
        GreenLabelMax = tk.Label(Frame, text="", bg="gray86")

        BlueEntryMin = tk.Entry(Frame, width=6)
        BlueEntryMax = tk.Entry(Frame, width=6)
        BlueLabelMin = tk.Label(Frame, text="", bg="gray86")
        BlueLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Saturation ####################################################

        SatEntryMin = tk.Entry(Frame, width=6)
        SatEntryMax = tk.Entry(Frame, width=6)
        SatLabelMin = tk.Label(Frame, text="", bg="gray86")
        SatLabelMax = tk.Label(Frame, text="", bg="gray86")

        ####################################### Graysvcale#########################################################

        GrayEntryMin = tk.Entry(Frame, width=6)
        GrayEntryMax = tk.Entry(Frame, width=6)
        GrayLabelMin = tk.Label(Frame, text="", bg="gray86")
        GrayLabelMax = tk.Label(Frame, text="", bg="gray86")

        ############################################# Temperature ##################################################

        TempEntryMin = tk.Entry(Frame, width=6)
        TempEntryMax = tk.Entry(Frame, width=6)
        TempLabelMin = tk.Label(Frame, text="", bg="gray86")
        TempLabelMax = tk.Label(Frame, text="", bg="gray86")

        ######################################### K Means Colour ###################################################

        KMeanEntryMin = tk.Entry(Frame, width=6)
        KMeanEntryMax = tk.Entry(Frame, width=6)
        KMeanLabelMin = tk.Label(Frame, text="", bg="gray86")
        KMeanLabelMax = tk.Label(Frame, text="", bg="gray86")

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        def DisplaySubOptions():
            # nonlocal CheckWidgetValues
            # nonlocal EntryWidgetValues

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Invert Colours":

                CheckWidgetValues.append(RedInvVar)
                CheckWidgetValues.append(GreenInvVar)
                CheckWidgetValues.append(BlueInvVar)

                SubDescriptionLabel.config(text="Select channels to invert", bg=DefaultCol)
                SubDescriptionLabel.place(relx=0.5, rely=0.2, anchor="center")

                RedInvert.config(text="Red")

                GreenInvert.config(text="Green")

                BlueInvert.config(text="Blue")

                RedInvert.place(relx=0.3, rely=0.3, anchor="center")
                GreenInvert.place(relx=0.5, rely=0.3, anchor="center")
                BlueInvert.place(relx=0.7, rely=0.3, anchor="center")
                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.append(RedInvert), ActiveWidgetList.append(GreenInvert), ActiveWidgetList.append(
                    BlueInvert)
                ActiveWidgetList.append(ConfirmButton)
                if SubOptionHistory[No] == "Invert Colours":
                    try:

                        if SavedWidgetValueList[No][0] == 1:
                            RedInvert.select()

                        if SavedWidgetValueList[No][1] == 1:
                            GreenInvert.select()

                        if SavedWidgetValueList[No][2] == 1:
                            BlueInvert.select()




                    except:
                        print("ERROR Invert Colours")

            elif Option == "RGB Levels":

                CheckWidgetValues.append(RedLevelVar)

                CheckWidgetValues.append(GreenLevelVar)

                CheckWidgetValues.append(BlueLevelVar)

                def ActivateRed():
                    RedEntryMin.config(state="disabled")
                    RedEntryMax.config(state="disabled")
                    RedLabelMin.config(text="Minimum:", fg="grey")
                    RedLabelMax.config(text="Maximum:", fg="grey")

                    if RedLevelVar.get() == 1:
                        RedEntryMin.config(state="normal")
                        RedEntryMax.config(state="normal")
                        RedLabelMin.config(text="Minimum:", fg="black")
                        RedLabelMax.config(text="Maximum:", fg="black")

                def ActivateGreen():
                    GreenEntryMin.config(state="disabled")
                    GreenEntryMax.config(state="disabled")
                    GreenLabelMin.config(text="Minimum:", fg="grey")
                    GreenLabelMax.config(text="Maximum:", fg="grey")

                    if GreenLevelVar.get() == 1:
                        GreenEntryMin.config(state="normal")
                        GreenEntryMax.config(state="normal")
                        GreenLabelMin.config(text="Minimum:", fg="black")
                        GreenLabelMax.config(text="Maximum:", fg="black")

                def ActivateBlue():
                    BlueEntryMin.config(state="disabled")
                    BlueEntryMax.config(state="disabled")
                    BlueLabelMin.config(text="Minimum:", fg="grey")
                    BlueLabelMax.config(text="Maximum:", fg="grey")

                    if BlueLevelVar.get() == 1:
                        BlueEntryMin.config(state="normal")
                        BlueEntryMax.config(state="normal")
                        BlueLabelMin.config(text="Minimum:", fg="black")
                        BlueLabelMax.config(text="Maximum:", fg="black")

                SubDescriptionLabel.config(text="Adjust the red green and blue pixel values.\n"
                                                "Please enter whole values between -255 and 255")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                RedLevelCheck.place(relx=0.5, rely=0.2, anchor="center")
                RedLevelCheck.config(text="Red", command=ActivateRed)
                RedLevelVar.set(1)
                RedLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                RedLabelMin.config(text="Minimum:")
                RedLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                RedLabelMax.config(text="Maximum:")
                RedEntryMin.place(relx=0.45, rely=0.27, anchor="center")  # Red min entry
                RedEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry

                GreenLevelCheck.place(relx=0.5, rely=0.4, anchor="center")
                GreenLevelCheck.config(text="Green", command=ActivateGreen)
                GreenLevelVar.set(1)
                GreenLabelMin.place(relx=0.3, rely=0.47, anchor="center")
                GreenLabelMin.config(text="Minimum:")
                GreenLabelMax.place(relx=0.6, rely=0.47, anchor="center")
                GreenLabelMax.config(text="Maximum:")
                GreenEntryMin.place(relx=0.45, rely=0.47, anchor="center")  # Green max entry
                GreenEntryMax.place(relx=0.75, rely=0.47, anchor="center")  # Green max entry

                BlueLevelCheck.place(relx=0.5, rely=0.6, anchor="center")
                BlueLevelCheck.config(text="Blue", command=ActivateBlue)
                BlueLevelVar.set(1)
                BlueLabelMin.place(relx=0.3, rely=0.67, anchor="center")
                BlueLabelMin.config(text="Minimum:")
                BlueLabelMax.place(relx=0.6, rely=0.67, anchor="center")
                BlueLabelMax.config(text="Maximum:")
                BlueEntryMin.place(relx=0.45, rely=0.67, anchor="center")  # Blue max entry
                BlueEntryMax.place(relx=0.75, rely=0.67, anchor="center")  # Blue max entry

                ConfirmButton.place(relx=0.5, rely=0.95, anchor="center")

                ActiveWidgetList.extend(
                    (RedLabelMin, RedLabelMax, GreenLabelMin, GreenLabelMax, BlueLabelMin, BlueLabelMax,
                     RedLevelCheck, GreenLevelCheck, BlueLevelCheck,
                     RedEntryMin, RedEntryMax, GreenEntryMin, GreenEntryMax, BlueEntryMin, BlueEntryMax,
                     ConfirmButton))

                print(f"True Sub {SubOptionHistory[No]}")
                if SubOptionHistory[No] == "RGB Levels":
                    try:
                        print(SavedWidgetValueList[No][0])
                        print(SavedWidgetValueList[No][1])
                        print(SavedWidgetValueList[No][2])

                        if SavedWidgetValueList[No][0] == 1:

                            RedLevelCheck.select()
                            RedEntryMin.insert(0, str(SavedWidgetValueList[No][3]))
                            RedEntryMax.insert(0, str(SavedWidgetValueList[No][4]))
                        else:
                            print("Red off")
                            RedLevelCheck.deselect()
                            RedLevelVar.set(0)
                            ActivateRed()
                            RedLevelCheck.deselect()

                        if SavedWidgetValueList[No][1] == 1:
                            GreenLevelCheck.select()
                            GreenEntryMin.insert(0, str(SavedWidgetValueList[No][5]))
                            GreenEntryMax.insert(0, str(SavedWidgetValueList[No][6]))
                        else:
                            GreenLevelCheck.deselect()
                            GreenLevelVar.set(0)
                            ActivateGreen()
                            GreenLevelCheck.deselect()

                        if SavedWidgetValueList[No][2] == 1:
                            BlueLevelCheck.select()
                            BlueEntryMin.insert(0, str(SavedWidgetValueList[No][7]))
                            BlueEntryMax.insert(0, str(SavedWidgetValueList[No][8]))
                        else:
                            BlueLevelCheck.deselect()
                            BlueLevelVar.set(0)
                            ActivateBlue()
                            BlueLevelCheck.deselect()
                    except:
                        print("NOT WORKING")

            elif Option == "Saturation":
                SubDescriptionLabel.config(text="Adjust the Saturation\n"
                                                "Enter whole values between -255 and 255.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                SatLabelMin.place(relx=0.30, rely=0.27, anchor="center")
                SatLabelMin.config(text="Minimum:")
                SatLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                SatLabelMax.config(text="Maximum:")
                SatEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                SatEntryMin.config(state="normal")
                SatEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                SatEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend((SatLabelMin, SatLabelMax, SatEntryMin, SatEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Saturation":
                    try:
                        SatEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        SatEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                    except:
                        print("Error with Saturation Settings")

            elif Option == "Greyscale":
                SubDescriptionLabel.config(text="Greyscale your images\n"
                                                "Enter a value between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                GrayLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                GrayLabelMin.config(text="Minimum:")
                GrayLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                GrayLabelMax.config(text="Maximum:")
                GrayEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                GrayEntryMax.place(relx=0.75, rely=0.27, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend((GrayLabelMin, GrayLabelMax, GrayEntryMin, GrayEntryMax, ConfirmButton))

                print(f"Grayvalues= {SavedWidgetValueList[No][0], SavedWidgetValueList[No][1]}")
                if SubOptionHistory[No] == "Greyscale":
                    try:
                        GrayEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        GrayEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                    except:
                        print("Error with Gray Settings")

            elif Option == "Temprature":
                SubDescriptionLabel.config(text="Adjust the Temperature of your images\n"
                                                "Enter a value between 1000 (hotter) and 40000 (colder)")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                TempLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                TempLabelMin.config(text="Minimum:")
                TempLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                TempLabelMax.config(text="Maximum:")
                TempEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                TempEntryMax.place(relx=0.75, rely=0.27, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend(
                    (TempLabelMin, TempLabelMax, TempEntryMin, TempEntryMax, ConfirmButton))

                if SubOptionHistory[No] == "Temprature":
                    try:
                        TempEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        TempEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                    except:
                        print("Error with Temp Settings")

            elif Option == "K-means Colour Quantization":
                SubDescriptionLabel.config(text="Pixels are grouped into K-clusters.\n"
                                                "Enter a whole positive numbers for the number of clusters.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                KMeanLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                KMeanLabelMin.config(text="Minimum:")
                KMeanLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                KMeanLabelMax.config(text="Maximum:")
                KMeanEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                KMeanEntryMax.place(relx=0.75, rely=0.27, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend(
                    (KMeanLabelMin, KMeanLabelMax, KMeanEntryMin, KMeanEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "K-means Colour Quantization":
                    try:
                        KMeanEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        KMeanEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                    except:
                        print("Error with KMean Settings")

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print("Thisisit",SavedWidgetValueList)

        if SubOptionHistory[Number] != "Select Colour adjustment option":
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set("Select Colour adjustment option")

        DisplaySubOptions()

        ###################################################################################################

    def DisplaySizeOptions(Number, Frame, LastSub):
        No = Number

        def ShowDropdown(WidgetVar, Widget):
            if WidgetVar.get() == 1:
                print("SHow")
                Widget.place(relx=0.5, rely=0.5, anchor="center")
            else:
                try:
                    Widget.place_forget()
                    print("Unshow")
                except:
                    pass

        SizeOptions = ["Resize", "Crop", "Pad"]
        SubTypeDropdown = ttk.Combobox(Frame, values=SizeOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())
        print("Should be Setting")
        SubTypeDropdown.set("Select Colour adjustment option")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Resize":
                AugList = []
                ResizeHeight = int(ResizeHeightEntry.get())
                ResizeWidth = int(ResizeWidthEntry.get())


                ImportantWidgets.extend((ResizeHeightEntry, ResizeWidthEntry))
                ImportantWidgetValues.extend((ResizeHeight, ResizeWidth))
                AugList.append(iaa.Resize({"height":ResizeHeight, "width": ResizeWidth}))

                try:
                    # del SavedOptionCommand[No]
                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetList.insert(No, ImportantWidgets)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

                print(SavedWidgetList)
                print("SavedWidgetValueList: ", SavedWidgetValueList)

            if SubTypeDropdown.get() == "Crop":
                AugList = []

                CropHeight = int(CropHeightEntry.get())
                CropWidth = int(CropWidthEntry.get())

                if FixedCropVar.get() == 1:
                    print("Fixed Crop Location Selected:" + FixedCropDropdown.get())
                    Crop_Location = FixedCropDropdown.get()
                    AugList.append(iaa.CropToFixedSize(width=CropWidth, height=CropHeight, position=Crop_Location))
                else:
                    AugList.append(iaa.CropToFixedSize(width=CropWidth, height=CropHeight))

                ImportantWidgets.extend((CropHeightEntry, CropWidthEntry,FixedCropVar, FixedCropDropdown ))
                ImportantWidgetValues.extend((CropHeight, CropWidth, FixedCropVar.get(), str(FixedCropDropdown.get())))


                try:
                    # del SavedOptionCommand[No]
                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetList.insert(No, ImportantWidgets)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Pad":
                AugList = []

                PadHeight = int(PadHeightEntry.get())
                PadWidth = int(PadWidthEntry.get())

                PadStyle = StylePadDropdown.get()




                if FixedPadVar.get() == 1:
                    Crop_Location = FixedPadDropdown.get()

                    if PadStyle == "Black":
                        PadStyle = "constant"
                        PadValue = 0
                        AugList.append(iaa.PadToFixedSize(width=PadWidth, height=PadHeight, position=Crop_Location,
                                                          pad_mode=PadStyle, pad_cval=PadValue))

                    elif PadStyle == "White":
                        PadStyle = "constant"
                        PadValue = 1
                        AugList.append(iaa.PadToFixedSize(width=PadWidth, height=PadHeight, position=Crop_Location,
                                                          pad_mode=PadStyle, pad_cval=PadValue))
                    else:
                        AugList.append(iaa.PadToFixedSize(width=PadWidth, height=PadHeight, position=Crop_Location,
                                                          pad_mode=PadStyle))



                else:
                    if PadStyle == "Black":
                        PadStyle = "constant"
                        PadValue = 0
                        AugList.append(iaa.PadToFixedSize(width=PadWidth, height=PadHeight, pad_mode=PadStyle,
                                                          pad_cval=PadValue))

                    elif PadStyle == "White":
                        PadStyle = "constant"
                        PadValue = 1
                        AugList.append(iaa.PadToFixedSize(width=PadWidth, height=PadHeight, pad_mode=PadStyle,
                                                          pad_cval=PadValue))

                    else:
                        AugList.append(iaa.PadToFixedSize(width=PadWidth, height=PadHeight,
                                                          pad_mode=PadStyle))

                ImportantWidgets.extend((PadHeightEntry, PadWidthEntry, FixedPadVar, FixedPadDropdown, StylePadDropdown))
                ImportantWidgetValues.extend(
                    (PadHeight, PadWidth, FixedPadVar.get(), str(FixedPadDropdown.get()), str(StylePadDropdown.get())))

                try:

                    del SavedWidgetValueList[No]
                    del SavedWidgetList[No]
                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass



        OptionsText.place_forget()

        ################################### Resize Widgets ################################################


        ResizeHeightEntry = tk.Entry(Frame, width=6)
        ResizeHeightLabel = tk.Label(Frame, text="Height (px)", bg="gray86")
        ResizeWidthEntry = tk.Entry(Frame, width=6)
        ResizeWidthLabel = tk.Label(Frame, text="Width (px)", bg="gray86")

        ############################## Crop Widgets ######################################################

        CropHeightEntry = tk.Entry(Frame, width=6)
        CropHeightLabel = tk.Label(Frame, text="Height (px)", bg="gray86")
        CropWidthEntry = tk.Entry(Frame, width=6)
        CropWidthLabel = tk.Label(Frame, text="Width (px)", bg="gray86")

        FixedCropVar = tk.IntVar()
        FixedCropCheck = tk.Checkbutton(Frame, text="Fixed Crop Position", variable=FixedCropVar, onvalue=1, offvalue=0, command=lambda: ShowDropdown(FixedCropVar, FixedCropDropdown), bg="gray86")


        FixedCropValues = ["center", "left-top", "left-center", "left-bottom", "center-top", "center-center", "center-bottom", "right-top", "right-center", "right-bottom"]
        FixedCropDropdown = ttk.Combobox(Frame, values=FixedCropValues, width=35, state="readonly")
        FixedCropDropdown.set("center")

        ############################## Pad Widgets #########################################################

        PadHeightEntry = tk.Entry(Frame, width=6)
        PadHeightLabel = tk.Label(Frame, text="Height (px)", bg="gray86")
        PadWidthEntry = tk.Entry(Frame, width=6)
        PadWidthLabel = tk.Label(Frame, text="Width (px)", bg="gray86")

        FixedPadVar = tk.IntVar()
        FixedPadCheck = tk.Checkbutton(Frame, text="Fixed Pad Position", variable=FixedPadVar, onvalue=1,
                                        offvalue=0, bg="gray86", command=lambda: ShowDropdown(FixedPadVar, FixedPadDropdown))

        OrigSizeVarPad = tk.IntVar()
        OrigSizeCheckPad = tk.Checkbutton(Frame, text="Keep Original Size", variable=OrigSizeVarPad, onvalue=1,
                                       offvalue=0, bg="gray86")

        FixedPadValues = ["center", "left-top", "left-center", "left-bottom", "center-top", "center-center",
                           "center-bottom", "right-top", "right-center", "right-bottom"]
        FixedPadDropdown = ttk.Combobox(Frame, values=FixedPadValues, width=35, state="readonly")
        FixedPadDropdown.set("center")

        PadStyleLabel = tk.Label(Frame, text="Select Padding Style:", bg="gray86")
        StylePadValues = ["Black", "White", "edge", "maximum", "median", "minimum", "reflect", "symmetric", "wrap"]
        StylePadDropdown = ttk.Combobox(Frame, values=StylePadValues, width=35, state="readonly")
        StylePadDropdown.set("Black")



        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)


        def DisplaySubOptions():
            SubDescriptionLabel.config(text="")

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Resize":
                SubDescriptionLabel.config(text="Resize your images")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ResizeHeightEntry.place(relx=0.4, rely=0.4, anchor="center")
                ResizeHeightLabel.place(relx=0.25, rely=0.4, anchor="center")
                ResizeWidthEntry.place(relx=0.7, rely=0.4, anchor="center")
                ResizeWidthLabel.place(relx=0.55, rely=0.4, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend(
                    (ResizeHeightLabel, ResizeHeightEntry, ResizeWidthLabel, ResizeWidthEntry, ConfirmButton))


                if SubOptionHistory[No] == "Resize":
                    try:
                         ResizeHeightEntry.insert(0,str(SavedWidgetValueList[No][0]))
                         ResizeWidthEntry.insert(0,str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error, Resize")

            if Option == "Crop":
                SubDescriptionLabel.config(text="Crop images without padding")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                CropHeightEntry.place(relx=0.4, rely=0.3, anchor="center")
                CropHeightLabel.place(relx=0.25, rely=0.3, anchor="center")
                CropWidthEntry.place(relx=0.7, rely=0.3, anchor="center")
                CropWidthLabel.place(relx=0.55, rely=0.3, anchor="center")



                FixedCropCheck.place(relx=0.5, rely=0.4, anchor="center")
                FixedCropDropdown.place(relx=0.5, rely=0.5, anchor="center")


                FixedCropVar.set(1)


                ConfirmButton.place(relx=0.5, rely=0.7, anchor="center")

                ActiveWidgetList.extend(
                    (CropHeightLabel, CropHeightEntry, CropWidthLabel, CropWidthEntry, FixedCropDropdown, FixedCropCheck,
                     ConfirmButton))

                if SubOptionHistory[No] == "Crop":
                    try:
                         CropHeightEntry.insert(0,str(SavedWidgetValueList[No][0]))
                         CropWidthEntry.insert(0,str(SavedWidgetValueList[No][1]))

                         if SavedWidgetValueList[No][2] == 1:
                             FixedCropVar.set(1)
                             FixedCropDropdown.set(str(SavedWidgetValueList[No][3]))


                    except:
                        print("Error, Crop")




            if Option == "Pad":
                SubDescriptionLabel.config(text="Pad images to constant size")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                PadHeightEntry.place(relx=0.4, rely=0.3, anchor="center")
                PadHeightLabel.place(relx=0.25, rely=0.3, anchor="center")
                PadWidthEntry.place(relx=0.7, rely=0.3, anchor="center")
                PadWidthLabel.place(relx=0.55, rely=0.3, anchor="center")

                FixedPadCheck.place(relx=0.5, rely=0.4, anchor="center")
                FixedPadDropdown.place(relx=0.5, rely=0.5, anchor="center")

                PadStyleLabel.place(relx=0.5, rely=0.6, anchor="center")
                StylePadDropdown.place(relx=0.5, rely=0.7, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.8, anchor="center")


                FixedPadVar.set(1)


                ActiveWidgetList.extend(
                    (PadHeightLabel, PadHeightEntry, PadWidthLabel, PadWidthEntry, FixedPadDropdown,
                     FixedPadCheck,
                     OrigSizeCheckPad, ConfirmButton, PadStyleLabel, StylePadDropdown))

                if SubOptionHistory[No] == "Pad":
                    try:
                        PadHeightEntry.insert(0, str(SavedWidgetValueList[No][0]))
                        PadWidthEntry.insert(0, str(SavedWidgetValueList[No][1]))

                        if SavedWidgetValueList[No][2] == 1:
                            FixedPadVar.set(1)
                            FixedPadDropdown.set(str(SavedWidgetValueList[No][3]))

                        StylePadDropdown.set(SavedWidgetValueList[No][4])


                    except:
                        print("Error, Crop")

        if SubOptionHistory[Number] != "Select Size adjustment option":
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set("Select Size adjustment option")

        DisplaySubOptions()

    def DisplayBriConOptions(Number, Frame, LastSub):
        No = Number
        ColourOptions = ["Brightness", "Gamma Contrast", "Sigmoid Contrast", "Log Contrast", "Linear Contrast",
                         "Contrast Limited Adaptive Histogram Equalization", "Histogram Equalisation"]
        SubTypeDropdown = ttk.Combobox(Frame, values=ColourOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Brightness and Contrast Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Brightness":
                AugList = []

                AugList.append(iaa.AddToBrightness((int(BriEntryMin.get()), int(BriEntryMax.get()))))
                ImportantWidgetValues.extend((int(BriEntryMin.get()), int(BriEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Gamma Contrast":
                AugList = []

                AugList.append(iaa.GammaContrast((float(GamConEntryMin.get()), float(GamConEntryMax.get()))))
                ImportantWidgetValues.extend((float(GamConEntryMin.get()), float(GamConEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Sigmoid Contrast":
                AugList = []

                AugList.append(iaa.SigmoidContrast(gain=(int(SigConEntryMinGain.get()), int(SigConEntryMaxGain.get())),
                                                   cutoff=(float(SigConEntryMinCutoff.get()),
                                                           float(SigConEntryMaxCutoff.get()))))
                ImportantWidgetValues.extend((float(SigConEntryMinCutoff.get()), float(SigConEntryMaxCutoff.get()),
                                              int(SigConEntryMinGain.get()), int(SigConEntryMaxGain.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Log Contrast":
                AugList = []

                AugList.append(iaa.LogContrast((float(LogConEntryMin.get()), float(LogConEntryMax.get()))))
                ImportantWidgetValues.extend((float(LogConEntryMin.get()), float(LogConEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Linear Contrast":
                AugList = []

                AugList.append(iaa.LinearContrast((float(LinConEntryMin.get()), float(LinConEntryMax.get()))))
                ImportantWidgetValues.extend((float(LinConEntryMin.get()), float(LinConEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Contrast Limited Adaptive Histogram Equalization":
                AugList = []

                AugList.append(iaa.AllChannelsCLAHE(clip_limit=(int(CLAHEEntryMin.get()), int(CLAHEEntryMax.get()))))
                ImportantWidgetValues.extend((int(CLAHEEntryMin.get()), int(CLAHEEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Histogram Equalisation":
                AugList = []

                AugList.append(iaa.HistogramEqualization())
                ImportantWidgetValues.append(0)  # Placeholder

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        DefaultCol = "gray86"

        OptionsText.place_forget()

        ################################################## Brightness Widgets ####################################

        RedInvVar = tk.IntVar()
        # CheckWidgetValues.append(RedInvVar)
        GreenInvVar = tk.IntVar()
        # CheckWidgetValues.append(GreenInvVar)
        BlueInvVar = tk.IntVar()
        # CheckWidgetValues.append(BlueInvVar)

        BriEntryMin = tk.Entry(Frame, width=6)
        BriEntryMax = tk.Entry(Frame, width=6)
        BriLabelMin = tk.Label(Frame, text="", bg="gray86")
        BriLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################## Gamma Contrast Widgets #################################################

        GamConEntryMin = tk.Entry(Frame, width=6)
        GamConEntryMax = tk.Entry(Frame, width=6)
        GamConLabelMin = tk.Label(Frame, text="", bg="gray86")
        GamConLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Sigma Contrast Widgets ####################################################

        SigConGainLable = tk.Label(Frame, text="Enter the gain. Use whole values between 1 and 10", bg="gray86")
        SigConEntryMinGain = tk.Entry(Frame, width=6)
        SigConEntryMaxGain = tk.Entry(Frame, width=6)
        SigConLabelMinGain = tk.Label(Frame, text="", bg="gray86")
        SigConLabelMaxGain = tk.Label(Frame, text="", bg="gray86")

        SigConCutoffLabel = tk.Label(Frame, text="Adjust cutoff value between 0 and 1", bg="gray86")
        SigConEntryMinCutoff = tk.Entry(Frame, width=6)
        SigConEntryMaxCutoff = tk.Entry(Frame, width=6)
        SigConLabelMinCutoff = tk.Label(Frame, text="", bg="gray86")
        SigConLabelMaxCutoff = tk.Label(Frame, text="", bg="gray86")

        ####################################### Log Contrast Widgets #########################################################

        LogConEntryMin = tk.Entry(Frame, width=6)
        LogConEntryMax = tk.Entry(Frame, width=6)
        LogConLabelMin = tk.Label(Frame, text="", bg="gray86")
        LogConLabelMax = tk.Label(Frame, text="", bg="gray86")

        ############################################# Linear Contrast Widgets ##################################################

        LinConEntryMin = tk.Entry(Frame, width=6)
        LinConEntryMax = tk.Entry(Frame, width=6)
        LinConLabelMin = tk.Label(Frame, text="", bg="gray86")
        LinConLabelMax = tk.Label(Frame, text="", bg="gray86")

        ######################################### CLAHE widgets ###################################################

        CLAHEEntryMin = tk.Entry(Frame, width=6)
        CLAHEEntryMax = tk.Entry(Frame, width=6)
        CLAHELabelMin = tk.Label(Frame, text="", bg="gray86")
        CLAHELabelMax = tk.Label(Frame, text="", bg="gray86")

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Brightness":

                SubDescriptionLabel.config(text="Adjust the Brightness\n"
                                                "Enter whole values between -255 and 255.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                BriLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                BriLabelMin.config(text="Minimum:")
                BriLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                BriLabelMax.config(text="Maximum:")
                BriEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                BriEntryMin.config(state="normal")
                BriEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                BriEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend(
                    (BriLabelMin, BriLabelMax, BriEntryMin, BriEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Brightness":

                    try:
                        BriEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        BriEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                    except:
                        print("Error with Brightness")

            elif Option == "Gamma Contrast":

                SubDescriptionLabel.config(text="Adjust Gamma Contrast\n"
                                                "Enter  values between 0.0 and 5.0.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                GamConLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                GamConLabelMin.config(text="Minimum:")
                GamConLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                GamConLabelMax.config(text="Maximum:")
                GamConEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                GamConEntryMin.config(state="normal")
                GamConEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                GamConEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend(
                    (GamConLabelMin, GamConLabelMax, GamConEntryMin, GamConEntryMax, ConfirmButton))

                try:
                    GamConEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                    GamConEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                except:
                    print("Error with Gam Con")

            elif Option == "Sigmoid Contrast":
                SubDescriptionLabel.config(text="Adjust the Sigmoid Contrast")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                SigConCutoffLabel.place(anchor="center", relx=0.5, rely=0.2)
                SigConLabelMinCutoff.place(relx=0.3, rely=0.27, anchor="center")
                SigConLabelMinCutoff.config(text="Minimum:")
                SigConLabelMaxCutoff.place(relx=0.6, rely=0.27, anchor="center")
                SigConLabelMaxCutoff.config(text="Maximum:")
                SigConEntryMinCutoff.place(relx=0.45, rely=0.27, anchor="center")
                SigConEntryMinCutoff.config(state="normal")
                SigConEntryMaxCutoff.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                SigConEntryMaxCutoff.config(state="normal")

                SigConGainLable.place(anchor="center", relx=0.5, rely=0.38)
                SigConLabelMinGain.place(relx=0.3, rely=0.45, anchor="center")
                SigConLabelMinGain.config(text="Minimum:")
                SigConLabelMaxGain.place(relx=0.6, rely=0.45, anchor="center")
                SigConLabelMaxGain.config(text="Maximum:")
                SigConEntryMinGain.place(relx=0.45, rely=0.45, anchor="center")
                SigConEntryMinGain.config(state="normal")
                SigConEntryMaxGain.place(relx=0.75, rely=0.45, anchor="center")  # Red max entry
                SigConEntryMaxGain.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (SigConLabelMinCutoff, SigConLabelMaxCutoff, SigConEntryMinCutoff, SigConEntryMaxCutoff,
                     SigConLabelMinGain, SigConLabelMaxGain, SigConEntryMinGain, SigConEntryMaxGain,
                     SigConGainLable, SigConCutoffLabel,
                     ConfirmButton))

                if SubOptionHistory[No] == "Sigmoid Contrast":
                    try:
                        SigConEntryMinCutoff.insert(0, str(SavedWidgetValueList[No][0]))
                        SigConEntryMaxCutoff.insert(0, str(SavedWidgetValueList[No][1]))
                        SigConEntryMinGain.insert(0, str(SavedWidgetValueList[No][2]))
                        SigConEntryMaxGain.insert(0, str(SavedWidgetValueList[No][3]))
                    except:
                        print("Error with Sig Con")

            elif Option == "Log Contrast":
                SubDescriptionLabel.config(text="Adjust Log Contrast\n"
                                                "Enter values between 0.0 and 5.0.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                LogConLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                LogConLabelMin.config(text="Minimum:")
                LogConLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                LogConLabelMax.config(text="Maximum:")
                LogConEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                LogConEntryMin.config(state="normal")
                LogConEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                LogConEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend(
                    (LogConLabelMin, LogConLabelMax, LogConEntryMin, LogConEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Log Contrast":

                    try:
                        LogConEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        LogConEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                    except:
                        print("Error with Log Con")

            elif Option == "Linear Contrast":
                SubDescriptionLabel.config(text="Adjust Linear Contrast\n"
                                                "Enter values between 0.0 and 5.0.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                LinConLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                LinConLabelMin.config(text="Minimum:")
                LinConLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                LinConLabelMax.config(text="Maximum:")
                LinConEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                LinConEntryMin.config(state="normal")
                LinConEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                LinConEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend(
                    (LinConLabelMin, LinConLabelMax, LinConEntryMin, LinConEntryMax, ConfirmButton))

                if SubOptionHistory[No] == "Linear Contrast":
                    try:
                        LinConEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        LinConEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                    except:
                        print("Error with LinCon")

            elif Option == "Contrast Limited Adaptive Histogram Equalization":
                SubDescriptionLabel.config(text="Adjust CLAHE Contrast\n"
                                                "Enter whole values greater than 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                CLAHELabelMin.place(relx=0.3, rely=0.27, anchor="center")
                CLAHELabelMin.config(text="Minimum:")
                CLAHELabelMax.place(relx=0.6, rely=0.27, anchor="center")
                CLAHELabelMax.config(text="Maximum:")
                CLAHEEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                CLAHEEntryMin.config(state="normal")
                CLAHEEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                CLAHEEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend(
                    (CLAHELabelMin, CLAHELabelMax, CLAHEEntryMin, CLAHEEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Contrast Limited Adaptive Histogram Equalization":
                    try:
                        CLAHEEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        CLAHEEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                    except:
                        print("Error with CLAHE")

            elif Option == "Histogram Equalisation":
                SubDescriptionLabel.config(text="Histogram Equalisation\n"
                                                "No further options available.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.5, anchor="center")

                ActiveWidgetList.extend((ConfirmButton))

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        if SubOptionHistory[Number] != 'Select Brightness and Contrast Options':
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set('Select Brightness and Contrast Options')

        DisplaySubOptions()

    def DisplaySharpEmbOptions(Number, Frame, LastSub):
        OptionsText.place_forget()
        No = Number
        NoiseOptions = ["Sharpness", "Embossing"]
        SubTypeDropdown = ttk.Combobox(Frame, values=NoiseOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Sharpness/Embossing Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Sharpness":
                AugList = []

                AugList.append(iaa.Sharpen(alpha=(float(SharpEntryMin.get()), float(SharpEntryMax.get())),
                                           lightness=(float(AlphaEntryMin.get()), float(AlphaEntryMax.get()))))
                ImportantWidgetValues.extend((float(SharpEntryMin.get()), float(SharpEntryMax.get()),
                                              float(AlphaEntryMin.get()), float(AlphaEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Embossing":
                AugList = []

                AugList.append(iaa.Emboss(strength=(float(EmbEntryMin.get()), float(EmbEntryMax.get())),
                                          alpha=(float(AlphaEmbEntryMin.get()), float(AlphaEmbEntryMax.get()))))
                ImportantWidgetValues.extend((float(EmbEntryMin.get()), float(EmbEntryMax.get()),
                                              float(AlphaEmbEntryMin.get()), float(AlphaEmbEntryMax.get())))
                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            try:
                del SubOptionHistory[No]
                # del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        ########################################### Sharpness Widgets ####################################################

        SharpLabel = tk.Label(Frame, text="Enter sharpness levels between 0 and 1", bg="gray86")
        SharpEntryMin = tk.Entry(Frame, width=6)
        SharpEntryMax = tk.Entry(Frame, width=6)
        SharpLabelMin = tk.Label(Frame, text="", bg="gray86")
        SharpLabelMax = tk.Label(Frame, text="", bg="gray86")

        AlphaLabel = tk.Label(Frame, text="Enter alpha levles between 0 and 1", bg="gray86")
        AlphaEntryMin = tk.Entry(Frame, width=6)
        AlphaEntryMax = tk.Entry(Frame, width=6)
        AlphaLabelMin = tk.Label(Frame, text="", bg="gray86")
        AlphaLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Emboss Widgets ####################################################

        EmbLabel = tk.Label(Frame, text="Enter emboss levels between 0 and 2", bg="gray86")
        EmbEntryMin = tk.Entry(Frame, width=6)
        EmbEntryMax = tk.Entry(Frame, width=6)
        EmbLabelMin = tk.Label(Frame, text="", bg="gray86")
        EmbLabelMax = tk.Label(Frame, text="", bg="gray86")

        AlphaEmbLabel = tk.Label(Frame, text="Enter alpha levles between 0 and 1", bg="gray86")
        AlphaEmbEntryMin = tk.Entry(Frame, width=6)
        AlphaEmbEntryMax = tk.Entry(Frame, width=6)
        AlphaEmbLabelMin = tk.Label(Frame, text="", bg="gray86")
        AlphaEmbLabelMax = tk.Label(Frame, text="", bg="gray86")

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Sharpness":

                SubDescriptionLabel.config(text="Adjust the Sharpness")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                SharpLabel.place(relx=0.5, rely=0.2, anchor="center")

                SharpLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                SharpLabelMin.config(text="Minimum:")
                SharpLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                SharpLabelMax.config(text="Maximum:")
                SharpEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                SharpEntryMin.config(state="normal")
                SharpEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                SharpEntryMax.config(state="normal")

                AlphaLabel.place(relx=0.5, rely=0.35, anchor="center")

                AlphaLabelMin.place(relx=0.3, rely=0.45, anchor="center")
                AlphaLabelMin.config(text="Minimum:")
                AlphaLabelMax.place(relx=0.6, rely=0.45, anchor="center")
                AlphaLabelMax.config(text="Maximum:")
                AlphaEntryMin.place(relx=0.45, rely=0.45, anchor="center")
                AlphaEntryMin.config(state="normal")
                AlphaEntryMax.place(relx=0.75, rely=0.45, anchor="center")  # Red max entry
                AlphaEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (SharpLabelMin, SharpLabelMax, SharpEntryMin, SharpEntryMax,
                     AlphaLabelMin, AlphaLabelMax, AlphaEntryMin, AlphaEntryMax,
                     SharpLabel, AlphaLabel,
                     ConfirmButton))
                if SubOptionHistory[No] == "Sharpness":
                    try:
                        SharpEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        SharpEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        AlphaEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        AlphaEntryMax.insert(0, str(SavedWidgetValueList[No][3]))
                    except:
                        print("Error with Sig Con")

            elif Option == "Embossing":

                SubDescriptionLabel.config(text="Emboss the image")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                EmbLabel.place(relx=0.5, rely=0.2, anchor="center")

                EmbLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                EmbLabelMin.config(text="Minimum:")
                EmbLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                EmbLabelMax.config(text="Maximum:")
                EmbEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                EmbEntryMin.config(state="normal")
                EmbEntryMax.place(relx=0.75, rely=0.27, anchor="center")  # Red max entry
                EmbEntryMax.config(state="normal")

                AlphaEmbLabel.place(relx=0.5, rely=0.38, anchor="center")

                AlphaEmbLabelMin.place(relx=0.3, rely=0.45, anchor="center")
                AlphaEmbLabelMin.config(text="Minimum:")
                AlphaEmbLabelMax.place(relx=0.6, rely=0.45, anchor="center")
                AlphaEmbLabelMax.config(text="Maximum:")
                AlphaEmbEntryMin.place(relx=0.45, rely=0.45, anchor="center")
                AlphaEmbEntryMin.config(state="normal")
                AlphaEmbEntryMax.place(relx=0.75, rely=0.45, anchor="center")  # Red max entry
                AlphaEmbEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (EmbLabelMin, EmbLabelMax, EmbEntryMin, EmbEntryMax,
                     AlphaEmbLabelMin, AlphaEmbLabelMax, AlphaEmbEntryMin, AlphaEmbEntryMax,
                     EmbLabel, AlphaEmbLabel,
                     ConfirmButton))

                if SubOptionHistory[No] == "Embossing":
                    try:
                        EmbEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        EmbEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        AlphaEmbEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        AlphaEmbEntryMax.insert(0, str(SavedWidgetValueList[No][3]))
                    except:
                        print("Error with Sig Con")

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        if SubOptionHistory[Number] != 'Select Sharpness/Embossing Options':
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set('Select Sharpness/Embossing Options')

        DisplaySubOptions()

    def DisplayNoiseOptions(Number, Frame, LastSub):
        OptionsText.place_forget()
        No = Number
        NoiseOptions = ["Gaussian Noise", "Laplace Noise", "Poisson Noise", "Salt and Pepper", "Impulse Noise",
                        "Jpeg compression",
                        "Solarize", "Shot Noise", "Speckle Noise"]
        SubTypeDropdown = ttk.Combobox(Frame, values=NoiseOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Noise Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Gaussian Noise":
                AugList = []

                if GaussColourVar.get() == 1:
                    Min = float(NoiseGaussEntryMin.get())
                    Max = float(NoiseGaussEntryMax.get())
                    AugList.append(iaa.AdditiveGaussianNoise(
                        scale=(Min * 255, Max * 255), per_channel=True))
                else:
                    AugList.append(iaa.AdditiveGaussianNoise(
                        scale=(float(NoiseGaussEntryMin.get()) * 255, float(NoiseGaussEntryMax.get()) * 255),
                        per_channel=False))

                ImportantWidgetValues.extend(
                    (GaussColourVar.get(), float(NoiseGaussEntryMin.get()), float(NoiseGaussEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                print("In Confirm Options")
                for c in SavedOptionCommand:
                    print(c)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)
                print(AugList)

            if SubTypeDropdown.get() == "Laplace Noise":
                AugList = []

                if LapColourVar.get() == 1:
                    AugList.append(iaa.AdditiveLaplaceNoise(
                        scale=((float(NoiseLapEntryMin.get()) * 255, float(NoiseLapEntryMax.get()) * 255)),
                        per_channel=True))
                else:
                    AugList.append(iaa.AdditiveLaplaceNoise(
                        scale=(float(NoiseLapEntryMin.get()) * 255, float(NoiseLapEntryMax.get()) * 255),
                        per_channel=False))

                ImportantWidgetValues.extend(
                    (LapColourVar.get(), float(NoiseLapEntryMin.get()), float(NoiseLapEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Poisson Noise":
                AugList = []

                if PoiColourVar.get() == 1:
                    AugList.append(iaa.AdditivePoissonNoise((int(NoisePoiEntryMin.get()), int(NoisePoiEntryMax.get())),
                                                            per_channel=True))
                else:
                    AugList.append(iaa.AdditivePoissonNoise((int(NoisePoiEntryMin.get()), int(NoisePoiEntryMax.get())),
                                                            per_channel=False))

                ImportantWidgetValues.extend(
                    (PoiColourVar.get(), float(NoisePoiEntryMin.get()), float(NoisePoiEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Salt and Pepper":
                AugList = []

                if SPCoarseVar.get() == 1:
                    AugList.append(
                        iaa.CoarseSaltAndPepper(random.uniform(float(SPEntryMin.get()), float(SPEntryMax.get())),
                                                size_percent=(0.01, 0.1)))
                else:
                    AugList.append(iaa.SaltAndPepper(random.uniform(float(SPEntryMin.get()), float(SPEntryMax.get()))))

                ImportantWidgetValues.extend((SPCoarseVar.get(), float(SPEntryMin.get()), float(SPEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Impulse Noise":
                AugList = []

                AugList.append(
                    iaa.ImpulseNoise(random.uniform(float(NoiseImpEntryMin.get()), float(NoiseImpEntryMax.get()))))

                ImportantWidgetValues.extend((float(NoiseImpEntryMin.get()), float(NoiseImpEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Jpeg compression":
                AugList = []

                AugList.append(iaa.JpegCompression(int(JPEGEntryMin.get()), int(JPEGEntryMax.get())))
                ImportantWidgetValues.extend((int(JPEGEntryMin.get()), int(JPEGEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Solarize":
                AugList = []

                AugList.append(iaa.Solarize(1.0, threshold=(int(SolEntryMin.get()), int(SolEntryMax.get()))))
                ImportantWidgetValues.extend((int(SolEntryMin.get()), int(SolEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Shot Noise":  # Needs Work
                AugList = []

                AugList.append(iaa.imgcorruptlike.ShotNoise(
                    severity=random.randint(int(ShotEntryMin.get()), int(ShotEntryMax.get()))))
                ImportantWidgetValues.extend((int(ShotEntryMin.get()), int(ShotEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Speckle Noise":  # Needs Work
                AugList = []

                AugList.append(iaa.imgcorruptlike.SpeckleNoise(
                    severity=random.randint(int(SpecEntryMin.get()), int(SpecEntryMax.get()))))
                ImportantWidgetValues.extend((int(SpecEntryMin.get()), int(SpecEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            try:
                del SubOptionHistory[No]
                # del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())
            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        ########################################### Gaussian Widgets ####################################################

        NoiseGaussEntryMin = tk.Entry(Frame, width=6)
        NoiseGaussEntryMax = tk.Entry(Frame, width=6)
        NoiseGaussLabelMin = tk.Label(Frame, text="", bg="gray86")
        NoiseGaussLabelMax = tk.Label(Frame, text="", bg="gray86")
        GaussColourVar = tk.IntVar()
        GaussColourCheck = tk.Checkbutton(Frame, variable=GaussColourVar, onvalue=1, offvalue=0, bg="gray86",
                                          text="Colour")

        ########################################### Lap Widgets ####################################################

        NoiseLapEntryMin = tk.Entry(Frame, width=6)
        NoiseLapEntryMax = tk.Entry(Frame, width=6)
        NoiseLapLabelMin = tk.Label(Frame, text="", bg="gray86")
        NoiseLapLabelMax = tk.Label(Frame, text="", bg="gray86")
        LapColourVar = tk.IntVar()
        LapColourCheck = tk.Checkbutton(Frame, variable=LapColourVar, onvalue=1, offvalue=0, bg="gray86", text="Colour")

        ########################################### Poisson Widgets ####################################################

        NoisePoiEntryMin = tk.Entry(Frame, width=6)
        NoisePoiEntryMax = tk.Entry(Frame, width=6)
        NoisePoiLabelMin = tk.Label(Frame, text="", bg="gray86")
        NoisePoiLabelMax = tk.Label(Frame, text="", bg="gray86")
        PoiColourVar = tk.IntVar()
        PoiColourCheck = tk.Checkbutton(Frame, variable=PoiColourVar, onvalue=1, offvalue=0, bg="gray86", text="Colour")

        ########################################### Salt and Peper Widgets ####################################################

        SPEntryMin = tk.Entry(Frame, width=6)
        SPEntryMax = tk.Entry(Frame, width=6)
        SPLabelMin = tk.Label(Frame, text="", bg="gray86")
        SPLabelMax = tk.Label(Frame, text="", bg="gray86")
        SPCoarseVar = tk.IntVar()
        SPCoarseCheck = tk.Checkbutton(Frame, variable=SPCoarseVar, onvalue=1, offvalue=0, bg="gray86", text="Coarse Noise")

        ########################################### Impulse Widgets ####################################################

        NoiseImpEntryMin = tk.Entry(Frame, width=6)
        NoiseImpEntryMax = tk.Entry(Frame, width=6)
        NoiseImpLabelMin = tk.Label(Frame, text="", bg="gray86")
        NoiseImpLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### JPEG Widgets ####################################################

        JPEGEntryMin = tk.Entry(Frame, width=6)
        JPEGEntryMax = tk.Entry(Frame, width=6)
        JPEGLabelMin = tk.Label(Frame, text="", bg="gray86")
        JPEGLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Solarise Widgets ####################################################

        SolEntryMin = tk.Entry(Frame, width=6)
        SolEntryMax = tk.Entry(Frame, width=6)
        SolLabelMin = tk.Label(Frame, text="", bg="gray86")
        SolLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Shot Widgets ####################################################

        ShotEntryMin = tk.Entry(Frame, width=6)
        ShotEntryMax = tk.Entry(Frame, width=6)
        ShotLabelMin = tk.Label(Frame, text="", bg="gray86")
        ShotLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Speckle Widgets ####################################################

        SpecEntryMin = tk.Entry(Frame, width=6)
        SpecEntryMax = tk.Entry(Frame, width=6)
        SpecLabelMin = tk.Label(Frame, text="", bg="gray86")
        SpecLabelMax = tk.Label(Frame, text="", bg="gray86")

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Gaussian Noise":
                SubDescriptionLabel.config(text="Add Gaussian Noise \n Enter values between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                NoiseGaussLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                NoiseGaussLabelMin.config(text="Minimum:")
                NoiseGaussLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                NoiseGaussLabelMax.config(text="Maximum:")
                NoiseGaussEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                NoiseGaussEntryMin.config(state="normal")
                NoiseGaussEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                NoiseGaussEntryMax.config(state="normal")

                GaussColourCheck.place(relx=0.5, rely=0.4, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (NoiseGaussLabelMin, NoiseGaussLabelMax, NoiseGaussEntryMin, NoiseGaussEntryMax, GaussColourCheck,
                     ConfirmButton))

                if SavedWidgetValueList[No][0] == 1:
                    GaussColourCheck.select()

                else:
                    GaussColourCheck.deselect()
                if SubOptionHistory[No] == "Gaussian Noise":
                    try:
                        NoiseGaussEntryMin.insert(0, str(SavedWidgetValueList[No][1]))
                        NoiseGaussEntryMax.insert(0, str(SavedWidgetValueList[No][2]))

                    except:
                        print("Error with Gauss Noise")

            if Option == "Laplace Noise":
                SubDescriptionLabel.config(text="Add Laplace Noise\n Enter values between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                NoiseLapLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                NoiseLapLabelMin.config(text="Minimum:")
                NoiseLapLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                NoiseLapLabelMax.config(text="Maximum:")
                NoiseLapEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                NoiseLapEntryMin.config(state="normal")
                NoiseLapEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                NoiseLapEntryMax.config(state="normal")

                LapColourCheck.place(relx=0.5, rely=0.4, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (NoiseLapLabelMin, NoiseLapLabelMax, NoiseLapEntryMin, NoiseLapEntryMax, LapColourCheck,
                     ConfirmButton))

                if SavedWidgetValueList[No][0] == 1:
                    LapColourCheck.select()

                else:
                    LapColourCheck.deselect()
                if SubOptionHistory[No] == "Laplace Noise":
                    try:
                        NoiseLapEntryMin.insert(0, str(SavedWidgetValueList[No][1]))
                        NoiseLapEntryMax.insert(0, str(SavedWidgetValueList[No][2]))

                    except:
                        print("Error with Lap Noise")

            if Option == "Poisson Noise":
                SubDescriptionLabel.config(text="Add Poisson Noise\n Enter whole values between 0 and 255")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                NoisePoiLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                NoisePoiLabelMin.config(text="Minimum:")
                NoisePoiLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                NoisePoiLabelMax.config(text="Maximum:")
                NoisePoiEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                NoisePoiEntryMin.config(state="normal")
                NoisePoiEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                NoisePoiEntryMax.config(state="normal")

                PoiColourCheck.place(relx=0.5, rely=0.4, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (NoisePoiLabelMin, NoisePoiLabelMax, NoisePoiEntryMin, NoisePoiEntryMax, PoiColourCheck,
                     ConfirmButton))

                if SavedWidgetValueList[No][0] == 1:
                    PoiColourCheck.select()

                else:
                    PoiColourCheck.deselect()
                if SubOptionHistory[No] == "Poisson Noise":
                    try:
                        NoisePoiEntryMin.insert(0, str(SavedWidgetValueList[No][1]))
                        NoisePoiEntryMax.insert(0, str(SavedWidgetValueList[No][2]))

                    except:
                        print("Error with Poi Noise")

            if Option == "Salt and Pepper":
                SubDescriptionLabel.config(text="Add Salt and Pepper Noise.\n Enter values between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                SPLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                SPLabelMin.config(text="Minimum:")
                SPLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                SPLabelMax.config(text="Maximum:")
                SPEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                SPEntryMin.config(state="normal")
                SPEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                SPEntryMax.config(state="normal")

                SPCoarseCheck.place(relx=0.5, rely=0.4, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (SPLabelMin, SPLabelMax, SPEntryMin, SPEntryMax, SPCoarseCheck,
                     ConfirmButton))

                if SavedWidgetValueList[No][0] == 1:
                    SPCoarseCheck.select()

                else:
                    SPCoarseCheck.deselect()
                if SubOptionHistory[No] == "Salt and Pepper":
                    try:
                        SPEntryMin.insert(0, str(SavedWidgetValueList[No][1]))
                        SPEntryMax.insert(0, str(SavedWidgetValueList[No][2]))

                    except:
                        print("Error with Poi Noise")

            if Option == "Impulse Noise":
                SubDescriptionLabel.config(text="Add Impulse Noise\n Enter values between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                NoiseImpLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                NoiseImpLabelMin.config(text="Minimum:")
                NoiseImpLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                NoiseImpLabelMax.config(text="Maximum:")
                NoiseImpEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                NoiseImpEntryMin.config(state="normal")
                NoiseImpEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                NoiseImpEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (NoiseImpLabelMin, NoiseImpLabelMax, NoiseImpEntryMin, NoiseImpEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Impulse Noise":
                    try:
                        NoiseImpEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        NoiseImpEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with imp Noise")

            if Option == "Jpeg compression":
                SubDescriptionLabel.config(text="Add JPEG comression effect.\n Enter strength values between 0 and 100")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                JPEGLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                JPEGLabelMin.config(text="Minimum:")
                JPEGLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                JPEGLabelMax.config(text="Maximum:")
                JPEGEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                JPEGEntryMin.config(state="normal")
                JPEGEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                JPEGEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (JPEGLabelMin, JPEGLabelMax, JPEGEntryMin, JPEGEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Jpeg Compression":
                    try:
                        JPEGEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        JPEGEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with JPEG Noise")

            if Option == "Solarize":
                SubDescriptionLabel.config(text="Inverts pixels with values above input values.\n"
                                                "Use whole values between 0 and 255")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                SolLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                SolLabelMin.config(text="Minimum:")
                SolLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                SolLabelMax.config(text="Maximum:")
                SolEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                SolEntryMin.config(state="normal")
                SolEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                SolEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (SolLabelMin, SolLabelMax, SolEntryMin, SolEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Solarize":
                    try:
                        SolEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        SolEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Sol Noise")

            if Option == "Shot Noise":
                SubDescriptionLabel.config(text="Add Shot Noise.\n Enter whole values between 0 and 5")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ShotLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                ShotLabelMin.config(text="Minimum:")
                ShotLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                ShotLabelMax.config(text="Maximum:")
                ShotEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                ShotEntryMin.config(state="normal")
                ShotEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                ShotEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (ShotLabelMin, ShotLabelMax, ShotEntryMin, ShotEntryMax,
                     ConfirmButton))

                if SubOptionHistory[No] == "Shot Noise":
                    try:
                        ShotEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        ShotEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Shot Noise")

            if Option == "Speckle Noise":

                SubDescriptionLabel.config(text="Add Speckle Noise\n Enter whole values between 0 and 5")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                SpecLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                SpecLabelMin.config(text="Minimum:")
                SpecLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                SpecLabelMax.config(text="Maximum:")
                SpecEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                SpecEntryMin.config(state="normal")
                SpecEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                SpecEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (SpecLabelMin, SpecLabelMax, SpecEntryMin, SpecEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Speckle Noise":
                    try:
                        SpecEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        SpecEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Spec Noise")

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        if SubOptionHistory[Number] != 'Select Noise Options':
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set('Select Noise Options')

        DisplaySubOptions()

    def DisplayDropoutOptions(Number, Frame, LastSub):
        OptionsText.place_forget()
        No = Number
        DropoutOptions = ["Cutout", "Dropout", "Coarse Dropout", "2D Dropout", "Total Dropout"]
        SubTypeDropdown = ttk.Combobox(Frame, values=DropoutOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Dropout Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Cutout":
                AugList = []

                AugList.append(iaa.Cutout(nb_iterations=(int(CutoutEntryMin.get()), int(CutoutEntryMax.get()))))

                ImportantWidgetValues.extend((int(CutoutEntryMin.get()), int(CutoutEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Dropout":
                AugList = []

                AugList.append(iaa.Dropout(p=(float(DropoutEntryMin.get()), float(DropoutEntryMax.get()))))

                ImportantWidgetValues.extend((float(DropoutEntryMin.get()), float(DropoutEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Coarse Dropout":
                AugList = []

                AugList.append(iaa.CoarseDropout((float(CorDropoutEntryMin.get()), float(CorDropoutEntryMax.get())),
                                                 size_percent=(float(CorDropoutSizeEntryMin.get()),
                                                               float(CorDropoutSizeEntryMax.get()))))

                ImportantWidgetValues.extend((float(CorDropoutEntryMin.get()), float(CorDropoutEntryMax.get()),
                                              float(CorDropoutSizeEntryMin.get()), float(CorDropoutSizeEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "2D Dropout":
                AugList = []

                AugList.append(iaa.Dropout2d(p=float(Dropout2DPercentageEntry.get())))

                ImportantWidgetValues.append(float(Dropout2DPercentageEntry.get()))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Total Dropout":
                AugList = []

                AugList.append(iaa.Dropout2d(p=float(DropoutTotalPercentageEntry.get())))

                ImportantWidgetValues.append(float(DropoutTotalPercentageEntry.get()))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            try:
                del SubOptionHistory[No]
                # del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        ########################################### Cutout Widgets ####################################################

        CutoutEntryMin = tk.Entry(Frame, width=6)
        CutoutEntryMax = tk.Entry(Frame, width=6)
        CutoutLabelMin = tk.Label(Frame, text="", bg="gray86")
        CutoutLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Dropout Widgets ####################################################

        DropoutEntryMin = tk.Entry(Frame, width=6)
        DropoutEntryMax = tk.Entry(Frame, width=6)
        DropoutLabelMin = tk.Label(Frame, text="", bg="gray86")
        DropoutLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Coarse Dropout Widgets ####################################################

        CorDropoutEntryMin = tk.Entry(Frame, width=6)
        CorDropoutEntryMax = tk.Entry(Frame, width=6)
        CorDropoutLabelMin = tk.Label(Frame, text="", bg="gray86")
        CorDropoutLabelMax = tk.Label(Frame, text="", bg="gray86")

        CorDropLabel = tk.Label(Frame, text="Enter relative size percentage. \n Use values between 0 and 1", bg="gray86")

        CorDropoutSizeEntryMin = tk.Entry(Frame, width=6)
        CorDropoutSizeEntryMax = tk.Entry(Frame, width=6)
        CorDropoutSizeLabelMin = tk.Label(Frame, text="", bg="gray86")
        CorDropoutSizeLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### 2D Dropout Widgets ####################################################

        Dropout2DPercentageEntry = tk.Entry(Frame, width=6)
        Dropout2DPercentageLabel = tk.Label(Frame, text="", bg="gray86")

        ########################################### 2D Dropout Widgets ####################################################

        DropoutTotalPercentageEntry = tk.Entry(Frame, width=6)
        DropoutTotalPercentageLabel = tk.Label(Frame, text="", bg="gray86")

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Cutout":
                SubDescriptionLabel.config(
                    text="Cut out random areas of each image.\n entered values are the number of areas\n"
                         "Enter whole numbers greater than 1.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                CutoutLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                CutoutLabelMin.config(text="Minimum:")
                CutoutLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                CutoutLabelMax.config(text="Maximum:")
                CutoutEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                CutoutEntryMin.config(state="normal")
                CutoutEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                CutoutEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (CutoutLabelMin, CutoutLabelMax, CutoutEntryMin, CutoutEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Cutout":
                    try:
                        CutoutEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        CutoutEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Cutout")

            if Option == "Dropout":
                SubDescriptionLabel.config(
                    text="Drop a percentage of pixels from images.\n Enter values between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                DropoutLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                DropoutLabelMin.config(text="Minimum:")
                DropoutLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                DropoutLabelMax.config(text="Maximum:")
                DropoutEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                DropoutEntryMin.config(state="normal")
                DropoutEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                DropoutEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (DropoutLabelMin, DropoutLabelMax, DropoutEntryMin, DropoutEntryMax,
                     ConfirmButton))

                if SubOptionHistory[No] == "Dropout":
                    try:
                        DropoutEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        DropoutEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Dropout")

            if Option == "Coarse Dropout":
                SubDescriptionLabel.config(
                    text="Cut out random areas of each image\n Enter percentage of pixels to remove\n"
                         "Use values between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                CorDropoutLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                CorDropoutLabelMin.config(text="Minimum:")
                CorDropoutLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                CorDropoutLabelMax.config(text="Maximum:")
                CorDropoutEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                CorDropoutEntryMin.config(state="normal")
                CorDropoutEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                CorDropoutEntryMax.config(state="normal")
                CorDropLabel.place(relx=0.5, rely=0.35, anchor="center")

                CorDropoutSizeLabelMin.place(relx=0.3, rely=0.45, anchor="center")
                CorDropoutSizeLabelMin.config(text="Width:")
                CorDropoutSizeLabelMax.place(relx=0.6, rely=0.45, anchor="center")
                CorDropoutSizeLabelMax.config(text="Height:")
                CorDropoutSizeEntryMin.place(relx=0.45, rely=0.45, anchor="center")
                CorDropoutSizeEntryMin.config(state="normal")
                CorDropoutSizeEntryMax.place(relx=0.75, rely=0.45, anchor="center")
                CorDropoutSizeEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (CorDropoutLabelMin, CorDropoutLabelMax, CorDropoutEntryMin, CorDropoutEntryMax,
                     CorDropoutSizeLabelMin, CorDropoutSizeLabelMax, CorDropoutSizeEntryMin, CorDropoutSizeEntryMax,
                     ConfirmButton, CorDropLabel))
                if SubOptionHistory[No] == "Coarse Dropout":
                    try:
                        CorDropoutEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        CorDropoutEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        CorDropoutSizeEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        CorDropoutSizeEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Dropout")

            if Option == "2D Dropout":
                SubDescriptionLabel.config(text="Drop random RGB channels from each image")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                Dropout2DPercentageLabel.place(relx=0.5, rely=0.33, anchor="center")
                Dropout2DPercentageLabel.config(text="Chance of channel dropout. enter values between 0 and 1")
                Dropout2DPercentageEntry.place(relx=0.5, rely=0.4, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (Dropout2DPercentageLabel,
                     Dropout2DPercentageEntry,
                     ConfirmButton))

                if SubOptionHistory[No] == "2D Dropout":
                    try:
                        Dropout2DPercentageEntry.insert(0, str(SavedWidgetValueList[No][0]))


                    except:
                        print("Error with 2dDropout")

            if Option == "Total Dropout":
                SubDescriptionLabel.config(text="Replace an entire image with black pixels")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                DropoutTotalPercentageLabel.place(relx=0.5, rely=0.33, anchor="center")
                DropoutTotalPercentageLabel.config(text="Chance of total dropout. enter values between 0 and 1")
                DropoutTotalPercentageEntry.place(relx=0.5, rely=0.4, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (DropoutTotalPercentageLabel,
                     DropoutTotalPercentageEntry,
                     ConfirmButton))
                if SubOptionHistory[No] == "Total Dropout":
                    try:
                        DropoutTotalPercentageEntry.insert(0, str(SavedWidgetValueList[No][0]))


                    except:
                        print("Error with 2dDropout")

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        if SubOptionHistory[Number] != 'Select Dropout Options':
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set('Select Dropout Options')

        DisplaySubOptions()

    def DisplayBlurOptions(Number, Frame, LastSub):
        OptionsText.place_forget()
        No = Number
        BlurOptions = ["Gaussian Blur", "Average Blur", "Median Blur", "Glass Blur", "Bilateral Blur", "Motion Blur"]
        SubTypeDropdown = ttk.Combobox(Frame, values=BlurOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Blur Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Gaussian Blur":
                AugList = []

                AugList.append(iaa.GaussianBlur(sigma=(float(GaussBlurEntryMin.get()), float(GaussBlurEntryMax.get()))))

                ImportantWidgetValues.extend((int(GaussBlurEntryMin.get()), int(GaussBlurEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Average Blur":
                AugList = []

                AugList.append(iaa.AverageBlur(k=((int(AvgBlurEntryMin.get()), int(AvgBlurEntryMax.get())),
                                                  (int(AvgBlurEntryMin.get()), int(AvgBlurEntryMax.get())))))

                ImportantWidgetValues.extend((int(AvgBlurEntryMin.get()), int(AvgBlurEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Median Blur":
                AugList = []

                AugList.append(iaa.MedianBlur(k=(int(MedBlurEntryMin.get()), int(MedBlurEntryMax.get()))))

                ImportantWidgetValues.extend((int(MedBlurEntryMin.get()), int(MedBlurEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Glass Blur":  # Something wrong distplaing thiss
                AugList = []

                intent = random.randint(int(GlassBlurEntryMin.get()), int(GlassBlurEntryMax.get()))
                AugList.append(iaa.imgcorruptlike.GlassBlur(severity=3))

                ImportantWidgetValues.extend((int(GlassBlurEntryMin.get()), int(GlassBlurEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Bilateral Blur":
                AugList = []

                AugList.append(
                    iaa.BilateralBlur(d=(int(BiBlurEntryMin.get()), int(BiBlurEntryMax.get())), sigma_color=(10, 250),
                                      sigma_space=(10, 250)))

                ImportantWidgetValues.extend((int(BiBlurEntryMin.get()), int(BiBlurEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Motion Blur":
                AugList = []
                anglelist = []

                for a in range(int(MotBlurDirEntryMin.get()), int(MotBlurDirEntryMax.get())):
                    anglelist.append(a)
                print(anglelist)
                Intensity = random.randint(int(MotBlurEntryMin.get()), int(MotBlurEntryMax.get()))
                AugList.append(iaa.MotionBlur(k=Intensity, angle=anglelist))

                ImportantWidgetValues.extend((int(MotBlurEntryMin.get()), int(MotBlurEntryMax.get()),
                                              int(MotBlurDirEntryMin.get()), int(MotBlurDirEntryMax.get())))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            try:
                del SubOptionHistory[No]
                # del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        ########################################### Gauss Blur Widgets ####################################################

        GaussBlurEntryMin = tk.Entry(Frame, width=6)
        GaussBlurEntryMax = tk.Entry(Frame, width=6)
        GaussBlurLabelMin = tk.Label(Frame, text="", bg="gray86")
        GaussBlurLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Average Blur Widgets ####################################################

        AvgBlurEntryMin = tk.Entry(Frame, width=6)
        AvgBlurEntryMax = tk.Entry(Frame, width=6)
        AvgBlurLabelMin = tk.Label(Frame, text="", bg="gray86")
        AvgBlurLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Median Blur Widgets ####################################################

        MedBlurEntryMin = tk.Entry(Frame, width=6)
        MedBlurEntryMax = tk.Entry(Frame, width=6)
        MedBlurLabelMin = tk.Label(Frame, text="", bg="gray86")
        MedBlurLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Glass Blur Widgets ####################################################

        GlassBlurEntryMin = tk.Entry(Frame, width=6)
        GlassBlurEntryMax = tk.Entry(Frame, width=6)
        GlassBlurLabelMin = tk.Label(Frame, text="", bg="gray86")
        GlassBlurLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Bilateral Blur Widgets ####################################################

        BiBlurEntryMin = tk.Entry(Frame, width=6)
        BiBlurEntryMax = tk.Entry(Frame, width=6)
        BiBlurLabelMin = tk.Label(Frame, text="", bg="gray86")
        BiBlurLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Bilateral Blur Widgets ####################################################

        BiBlurEntryMin = tk.Entry(Frame, width=6)
        BiBlurEntryMax = tk.Entry(Frame, width=6)
        BiBlurLabelMin = tk.Label(Frame, text="", bg="gray86")
        BiBlurLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Motion Blur Widgets ####################################################

        MotBlurEntryMin = tk.Entry(Frame, width=6)
        MotBlurEntryMax = tk.Entry(Frame, width=6)
        MotBlurLabelMin = tk.Label(Frame, text="", bg="gray86")
        MotBlurLabelMax = tk.Label(Frame, text="", bg="gray86")

        MotBlurDirEntryMin = tk.Entry(Frame, width=6)
        MotBlurDirEntryMax = tk.Entry(Frame, width=6)
        MotBlurDirLabelMin = tk.Label(Frame, text="", bg="gray86")
        MotBlurDirLabelMax = tk.Label(Frame, text="", bg="gray86")

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Gaussian Blur":
                SubDescriptionLabel.config(
                    text="Applies a gaussian filter.\n Enter kernal size using whole numbers greater than 1.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                GaussBlurLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                GaussBlurLabelMin.config(text="Minimum:")
                GaussBlurLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                GaussBlurLabelMax.config(text="Maximum:")
                GaussBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                GaussBlurEntryMin.config(state="normal")
                GaussBlurEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                GaussBlurEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (GaussBlurLabelMin, GaussBlurLabelMax, GaussBlurEntryMin, GaussBlurEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Gaussian Blur":
                    try:
                        GaussBlurEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        GaussBlurEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Gauss Blur Noise")

            if Option == "Average Blur":
                SubDescriptionLabel.config(
                    text="Average Blur over a neibhorhood of pixels.\n Enter whole positive values greater than 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                AvgBlurLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                AvgBlurLabelMin.config(text="Minimum:")
                AvgBlurLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                AvgBlurLabelMax.config(text="Maximum:")
                AvgBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                AvgBlurEntryMin.config(state="normal")
                AvgBlurEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                AvgBlurEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (AvgBlurLabelMin, AvgBlurLabelMax, AvgBlurEntryMin, AvgBlurEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Average Blur":
                    try:
                        AvgBlurEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        AvgBlurEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Avg Blur Noise")

            if Option == "Median Blur":
                SubDescriptionLabel.config(text="Apply a medain blur over a neigboirhood of pixels.\n"
                                                "Enter whole positive values greater than 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                MedBlurLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                MedBlurLabelMin.config(text="Minimum:")
                MedBlurLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                MedBlurLabelMax.config(text="Maximum:")
                MedBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                MedBlurEntryMin.config(state="normal")
                MedBlurEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                MedBlurEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (MedBlurLabelMin, MedBlurLabelMax, MedBlurEntryMin, MedBlurEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Median Blur":
                    try:
                        MedBlurEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        MedBlurEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Med Blur Noise")

            if Option == "Bilateral Blur":
                SubDescriptionLabel.config(text="Apply bilateral blur which tries to preserve edges.\n"
                                                "Enter Whole positive values")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                BiBlurLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                BiBlurLabelMin.config(text="Minimum:")
                BiBlurLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                BiBlurLabelMax.config(text="Maximum:")
                BiBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                BiBlurEntryMin.config(state="normal")
                BiBlurEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                BiBlurEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (BiBlurLabelMin, BiBlurLabelMax, BiBlurEntryMin, BiBlurEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Bilateral Blur":
                    try:
                        BiBlurEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        BiBlurEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Bi Blur Noise")

            if Option == "Glass Blur":
                SubDescriptionLabel.config(text="Apply a frosted glass effect.\n"
                                                "Enter whole values between 0 and 5")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                GlassBlurLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                GlassBlurLabelMin.config(text="Minimum:")
                GlassBlurLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                GlassBlurLabelMax.config(text="Maximum:")
                GlassBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                GlassBlurEntryMin.config(state="normal")
                GlassBlurEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                GlassBlurEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                if SubOptionHistory[No] == "Glass Blur":
                    try:
                        GlassBlurEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        GlassBlurEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        print("Error with Glass Blur Noise")

                ActiveWidgetList.extend(
                    (GlassBlurLabelMin, GlassBlurLabelMax, GlassBlurEntryMin, GlassBlurEntryMax,
                     ConfirmButton))

            if Option == "Motion Blur":
                SubDescriptionLabel.config(
                    text="Simulate motion blur on an image.\n Adjust intensity: Insert positive whole values.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                MotBlurLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                MotBlurLabelMin.config(text="Minimum:")
                MotBlurLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                MotBlurLabelMax.config(text="Maximum:")
                MotBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                MotBlurEntryMin.config(state="normal")
                MotBlurEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                MotBlurEntryMax.config(state="normal")

                MotBlurDirLabel = tk.Label(Frame, text="Enter a range of directions for blur (0-360)", bg="gray86")
                MotBlurDirLabel.place(relx=0.5, rely=0.35, anchor="center")

                MotBlurDirLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                MotBlurDirLabelMin.config(text="Minimum:")
                MotBlurDirLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                MotBlurDirLabelMax.config(text="Maximum:")
                MotBlurDirEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                MotBlurDirEntryMin.config(state="normal")
                MotBlurDirEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                MotBlurDirEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (MotBlurDirLabelMin, MotBlurDirLabelMax, MotBlurDirLabel, MotBlurLabelMin, MotBlurLabelMax,
                     MotBlurEntryMin, MotBlurEntryMax, MotBlurDirEntryMax, MotBlurDirEntryMin, ConfirmButton))
                if SubOptionHistory[No] == "Motion Blur":
                    try:
                        MotBlurEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        MotBlurEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        MotBlurDirEntryMax.insert(0, str(SavedWidgetValueList[No][2]))
                        MotBlurDirEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Motion Blur Noise")

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        if SubOptionHistory[Number] != 'Select Blur Options':
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set('Select Blur Options')

        DisplaySubOptions()

    def DisplayGeoOptions(Number, Frame, LastSub):
        OptionsText.place_forget()
        No = Number
        GeoOptions = ["Scale", "Translate", "Rotate", "Shear", "Piecewise Affine",
                      "Elastic Transformation", "Polar Warp", "Jigsaw"]
        SubTypeDropdown = ttk.Combobox(Frame, values=GeoOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Transformation Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Translate":
                AugList = []

                XMin = float(TransXEntryMin.get())
                XMax = float(TransXEntryMax.get())
                YMin = float(TransYEntryMin.get())
                YMax = float(TransYEntryMax.get())

                AugList.append(iaa.Affine(translate_percent={"x": (XMin, XMax), "y": (YMin, YMax)}))

                ImportantWidgetValues.extend((XMin, XMax, YMin, YMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Scale":
                AugList = []

                XMin = float(XscaleEntryMin.get())
                XMax = float(XscaleEntryMax.get())
                YMin = float(YscaleEntryMin.get())
                YMax = float(YscaleEntryMax.get())

                AugList.append(iaa.Affine(scale={"x": (XMin, XMax), "y": (YMin, YMax)}))

                ImportantWidgetValues.extend((XMin, XMax, YMin, YMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Rotate":
                AugList = []

                RotMin = int(RotEntryMin.get())
                RotMax = int(RotEntryMax.get())

                AugList.append(iaa.Affine(rotate=(RotMin, RotMax)))

                ImportantWidgetValues.extend((RotMin, RotMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Shear":
                AugList = []

                print(XscaleEntryMin.get(), XscaleEntryMax.get(), YscaleEntryMin.get)

                XMin = float(XshearEntryMin.get())
                XMax = float(XshearEntryMax.get())
                YMin = float(YshearEntryMin.get())
                YMax = float(YshearEntryMax.get())

                AugList.append(iaa.Affine(shear={"x": (XMin, XMax), "y": (YMin, YMax)}))

                ImportantWidgetValues.extend((XMin, XMax, YMin, YMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Piecewise Affine":
                AugList = []

                PAMin = float(PAEntryMin.get())
                PAMax = float(PAEntryMax.get())

                AugList.append(iaa.PiecewiseAffine(scale=(PAMin, PAMax)))

                ImportantWidgetValues.extend((PAMin, PAMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Elastic Transformation":
                AugList = []

                ETMin = float(ETEntryMin.get())
                ETMax = float(ETEntryMax.get())
                ETSig = float(ETEntrySigma.get())

                AugList.append(iaa.ElasticTransformation(alpha=(ETMin, ETMax), sigma=ETSig))

                ImportantWidgetValues.extend((ETMin, ETMax, ETSig))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Polar Warp":
                AugList = []

                polMin = float(polarEntryMin.get())
                polMax = float(polarEntryMax.get())

                if polarDropdown.get() == "Crop and Pad":
                    AugList.append(iaa.WithPolarWarping(iaa.CropAndPad(percent=(polMin, polMax))))
                if polarDropdown.get() == "Affine":
                    AugList.append(iaa.WithPolarWarping(
                        iaa.Affine(translate_percent={"x": (polMin, polMax), "y": (polMin, polMax)})))
                if polarDropdown.get() == "Average Pooling":
                    AugList.append(iaa.WithPolarWarping(iaa.AveragePooling((int(polMin), int(polMax)))))

                ImportantWidgetValues.extend((polMin, polMax, polarDropdown.get()))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Jigsaw":
                AugList = []

                Jigrowmin = int(JigRowsEntryMin.get())
                Jigrowmax = int(JigRowsEntryMax.get())
                Jigcolmin = int(JigColsEntryMin.get())
                Jigcolmax = int(JigColsEntryMax.get())

                AugList.append(iaa.Jigsaw(nb_rows=random.randint(Jigrowmin, Jigrowmax),
                                          nb_cols=random.randint(Jigcolmin, Jigcolmax)))

                ImportantWidgetValues.extend((Jigrowmin, Jigrowmax, Jigcolmin, Jigcolmax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            try:
                del SubOptionHistory[No]
                # del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        ########################################### Translate Widgets ####################################################

        TransXEntryMin = tk.Entry(Frame, width=6)
        TransXEntryMax = tk.Entry(Frame, width=6)
        TransXLabelMin = tk.Label(Frame, text="", bg="gray86")
        TransXLabelMax = tk.Label(Frame, text="", bg="gray86")

        TransYEntryMin = tk.Entry(Frame, width=6)
        TransYEntryMax = tk.Entry(Frame, width=6)
        TransYLabelMin = tk.Label(Frame, text="", bg="gray86")
        TransYLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Scale Widgets ####################################################

        XscaleEntryMin = tk.Entry(Frame, width=6)
        XscaleEntryMax = tk.Entry(Frame, width=6)
        XscaleLabelMin = tk.Label(Frame, text="", bg="gray86")
        XscaleLabelMax = tk.Label(Frame, text="", bg="gray86")

        YscaleEntryMin = tk.Entry(Frame, width=6)
        YscaleEntryMax = tk.Entry(Frame, width=6)
        YscaleLabelMin = tk.Label(Frame, text="", bg="gray86")
        YscaleLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Rotate Widgets ####################################################

        RotEntryMin = tk.Entry(Frame, width=6)
        RotEntryMax = tk.Entry(Frame, width=6)
        RotLabelMin = tk.Label(Frame, text="", bg="gray86")
        RotLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Shear Widgets ####################################################

        XshearEntryMin = tk.Entry(Frame, width=6)
        XshearEntryMax = tk.Entry(Frame, width=6)
        XshearLabelMin = tk.Label(Frame, text="", bg="gray86")
        XshearLabelMax = tk.Label(Frame, text="", bg="gray86")

        YshearEntryMin = tk.Entry(Frame, width=6)
        YshearEntryMax = tk.Entry(Frame, width=6)
        YshearLabelMin = tk.Label(Frame, text="", bg="gray86")
        YshearLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Piecewise Affine Widgets ####################################################

        PAEntryMin = tk.Entry(Frame, width=6)
        PAEntryMax = tk.Entry(Frame, width=6)
        PALabelMin = tk.Label(Frame, text="", bg="gray86")
        PALabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Elastic transform Widgets ####################################################

        ETEntryMin = tk.Entry(Frame, width=6)
        ETEntryMax = tk.Entry(Frame, width=6)
        ETEntrySigma = tk.Entry(Frame, width=6)
        ETLabelMin = tk.Label(Frame, text="", bg="gray86")
        ETLabelMax = tk.Label(Frame, text="", bg="gray86")
        ETLabelSigma = tk.Label(Frame, text="Sigma", bg="gray86")

        ########################################### Polar Warping Widgets ####################################################

        polarEntryMin = tk.Entry(Frame, width=6)
        polarEntryMax = tk.Entry(Frame, width=6)
        polarLabelMin = tk.Label(Frame, text="", bg="gray86")
        polarLabelMax = tk.Label(Frame, text="", bg="gray86")

        polarWarpOptions = ["Crop and Pad", "Affine", "Average Pooling"]
        polarDropdown = ttk.Combobox(Frame, values=polarWarpOptions, width=35, state="readonly",
                                     name=f"dropdownpolar{AugOptions}")
        polarDropdown.set("Select Warp Type")

        ########################################### Jigsaw Widgets ####################################################

        JigRowsEntryMin = tk.Entry(Frame, width=6)
        JigRowsEntryMax = tk.Entry(Frame, width=6)
        JigRowsLabelMin = tk.Label(Frame, text="", bg="gray86")
        JigRowsLabelMax = tk.Label(Frame, text="", bg="gray86")

        JigColsEntryMin = tk.Entry(Frame, width=6)
        JigColsEntryMax = tk.Entry(Frame, width=6)
        JigColsLabelMin = tk.Label(Frame, text="", bg="gray86")
        JigColsLabelMax = tk.Label(Frame, text="", bg="gray86")

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Scale":
                SubDescriptionLabel.config(
                    text="Scale the image on the X and Y axis\n Enter values greater than 0 (1 keeps the image the same).")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                XscaleLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                XscaleLabelMin.config(text="Minimum X:")
                XscaleLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                XscaleLabelMax.config(text="Maximum X:")
                XscaleEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                XscaleEntryMin.config(state="normal")
                XscaleEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                XscaleEntryMax.config(state="normal")

                YscaleLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                YscaleLabelMin.config(text="Minimum Y:")
                YscaleLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                YscaleLabelMax.config(text="Maximum Y:")
                YscaleEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                YscaleEntryMin.config(state="normal")
                YscaleEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                YscaleEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (XscaleLabelMin, XscaleLabelMax, XscaleEntryMin, XscaleEntryMax,
                     YscaleLabelMin, YscaleLabelMax, YscaleEntryMin, YscaleEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Scale":
                    try:
                        XscaleEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        XscaleEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        YscaleEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        YscaleEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Scale")

            if Option == "Translate":
                SubDescriptionLabel.config(
                    text="Translate the image along the x and y axis a a percentage of the image size.\n Enter any value (0 is default).")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                TransXLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                TransXLabelMin.config(text="Minimum X:")
                TransXLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                TransXLabelMax.config(text="Maximum X:")
                TransXEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                TransXEntryMin.config(state="normal")
                TransXEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                TransXEntryMax.config(state="normal")

                TransYLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                TransYLabelMin.config(text="Minimum Y:")
                TransYLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                TransYLabelMax.config(text="Maximum Y:")
                TransYEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                TransYEntryMin.config(state="normal")
                TransYEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                TransYEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (TransXLabelMin, TransXLabelMax, TransXEntryMin, TransXEntryMax,
                     TransYLabelMin, TransYLabelMax, TransYEntryMin, TransYEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Translate":
                    try:
                        TransXEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        TransXEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        TransYEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        TransYEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Trans")

            if Option == "Rotate":
                SubDescriptionLabel.config(text="Rotate images.\n Enter values between 0 and 360.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                RotLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                RotLabelMin.config(text="Minimum:")
                RotLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                RotLabelMax.config(text="Maximum:")
                RotEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                RotEntryMin.config(state="normal")
                RotEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                RotEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (RotLabelMin, RotLabelMax, RotEntryMin, RotEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Rotate":
                    try:
                        RotEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        RotEntryMax.insert(0, str(SavedWidgetValueList[No][1]))


                    except:
                        print("Error with Rots")

            if Option == "Shear":
                SubDescriptionLabel.config(
                    text="Add a shear (slanted) effect to the images.\n Enter values between 0 and 90")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                XshearLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                XshearLabelMin.config(text="Minimum X:")
                XshearLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                XshearLabelMax.config(text="Maximum X:")
                XshearEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                XshearEntryMin.config(state="normal")
                XshearEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                XshearEntryMax.config(state="normal")

                YshearLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                YshearLabelMin.config(text="Minimum Y:")
                YshearLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                YshearLabelMax.config(text="Maximum Y:")
                YshearEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                YshearEntryMin.config(state="normal")
                YshearEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                YshearEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (XshearLabelMin, XshearLabelMax, XshearEntryMin, XshearEntryMax,
                     YshearLabelMin, YshearLabelMax, YshearEntryMin, YshearEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Shear":
                    try:
                        XshearEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        XshearEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        YshearEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        YshearEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Shear")

            if Option == "Piecewise Affine":
                SubDescriptionLabel.config(
                    text="Causes random tranformations around local pixel clusters.\n Enter values between 0 and 0.5.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                PALabelMin.place(relx=0.3, rely=0.27, anchor="center")
                PALabelMin.config(text="Minimum:")
                PALabelMax.place(relx=0.6, rely=0.27, anchor="center")
                PALabelMax.config(text="Maximum:")
                PAEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                PAEntryMin.config(state="normal")
                PAEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                PAEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (PALabelMin, PALabelMax, PAEntryMin, PAEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Piecewise Affine":
                    try:
                        PAEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        PAEntryMax.insert(0, str(SavedWidgetValueList[No][1]))


                    except:
                        print("Error with PA")

            if Option == "Elastic Transformation":
                SubDescriptionLabel.config(
                    text="Displaces pixels around local clusters.\n Enter values between 0 and 10.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ETLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                ETLabelMin.config(text="Minimum:")
                ETLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                ETLabelMax.config(text="Maximum:")
                ETLabelSigma.place(relx=0.5, rely=0.35, anchor="center")
                ETLabelSigma.config(text="Sigma: enter values between 0 and 1")
                ETEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                ETEntryMin.config(state="normal")
                ETEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                ETEntryMax.config(state="normal")
                ETEntrySigma.place(relx=0.5, rely=0.43, anchor="center")
                ETEntrySigma.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (ETLabelMin, ETLabelMax, ETEntryMin, ETEntryMax, ETLabelSigma, ETEntrySigma, ConfirmButton))
                if SubOptionHistory[No] == "Elastic Transformation":
                    try:
                        ETEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        ETEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        ETEntrySigma.insert(0, str(SavedWidgetValueList[No][2]))


                    except:
                        print("Error with Shear")

            if Option == "Polar Warp":
                SubDescriptionLabel.config(text="Applies Transformations ancored around the center point of the image.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                polarLabelMin.place(relx=0.3, rely=0.4, anchor="center")
                polarLabelMin.config(text="Minimum:")
                polarLabelMax.place(relx=0.6, rely=0.4, anchor="center")
                polarLabelMax.config(text="Maximum:")
                polarEntryMin.place(relx=0.45, rely=0.4, anchor="center")
                polarEntryMin.config(state="normal")
                polarEntryMax.place(relx=0.75, rely=0.4, anchor="center")
                polarEntryMax.config(state="normal")

                polarDropdown.place(relx=0.5, rely=0.27, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (polarLabelMin, polarLabelMax, polarEntryMin, polarEntryMax, polarDropdown, ConfirmButton))
                if SubOptionHistory[No] == "Polar Warp":
                    try:
                        polarEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        polarEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        polarDropdown.insert(0, str(SavedWidgetValueList[No][2]))


                    except:
                        print("Error with Shear")

            if Option == "Jigsaw":
                SubDescriptionLabel.config(
                    text="Add a shear (slanted) effect to the images.\n Enter values between 0 and 90")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                JigRowsLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                JigRowsLabelMin.config(text="Rows Min:")
                JigRowsLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                JigRowsLabelMax.config(text="Rows Max:")
                JigRowsEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                JigRowsEntryMin.config(state="normal")
                JigRowsEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                JigRowsEntryMax.config(state="normal")

                JigColsLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                JigColsLabelMin.config(text="Columns Min:")
                JigColsLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                JigColsLabelMax.config(text="Columns Max:")
                JigColsEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                JigColsEntryMin.config(state="normal")
                JigColsEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                JigColsEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (JigRowsLabelMin, JigRowsLabelMax, JigRowsEntryMin, JigRowsEntryMax,
                     JigColsLabelMin, JigColsLabelMax, JigColsEntryMin, JigColsEntryMax,
                     ConfirmButton))

                if SubOptionHistory[No] == "Jigsaw":
                    try:
                        JigRowsEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        JigRowsEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        JigColsEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        JigColsEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Shear")

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        if SubOptionHistory[Number] != 'Select Transformation Options':
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set('Select Transformation Options')

        DisplaySubOptions()

    def DisplayEdgeOptions(Number, Frame, LastSub):
        OptionsText.place_forget()
        No = Number
        EdgeOptions = ["Canny Edge Detection", "Directional Edge Detection"]
        SubTypeDropdown = ttk.Combobox(Frame, values=EdgeOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Edge Detection Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Canny Edge Detection":
                AugList = []

                AMin = float(CannyEntryMin.get())
                AMax = float(CannyEntryMax.get())

                AugList.append(iaa.Canny(alpha=(AMin, AMax),
                                         colorizer=iaa.RandomColorsBinaryImageColorizer(color_true=255, color_false=0)))

                ImportantWidgetValues.extend((AMin, AMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Directional Edge Detection":
                AugList = []

                AMin = float(DirEdgeAlphaEntryMin.get())
                AMax = float(DirEdgeAlphaEntryMax.get())
                DirMin = float(DirEdgeEntryMin.get()) / 360
                DirMax = float(DirEdgeEntryMax.get()) / 360

                AugList.append(iaa.DirectedEdgeDetect(alpha=(AMin, AMax), direction=(DirMin, DirMax)))

                ImportantWidgetValues.extend((AMin, AMax, DirMin, DirMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            try:
                del SubOptionHistory[No]
                # del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        ########################################### Canny Widgets ####################################################

        CannyEntryMin = tk.Entry(Frame, width=6)
        CannyEntryMax = tk.Entry(Frame, width=6)
        CannyLabelMin = tk.Label(Frame, text="", bg="gray86")
        CannyLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Direction Edge Widgets ####################################################

        DirEdgeEntryMin = tk.Entry(Frame, width=6)
        DirEdgeEntryMax = tk.Entry(Frame, width=6)
        DirEdgeLabelMin = tk.Label(Frame, text="", bg="gray86")
        DirEdgeLabelMax = tk.Label(Frame, text="", bg="gray86")

        DirEdgeAlphaEntryMin = tk.Entry(Frame, width=6)
        DirEdgeAlphaEntryMax = tk.Entry(Frame, width=6)
        DirEdgeAlphaLabelMin = tk.Label(Frame, text="", bg="gray86")
        DirEdgeAlphaLabelMax = tk.Label(Frame, text="", bg="gray86")

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Canny Edge Detection":
                SubDescriptionLabel.config(
                    text="Canny Edge Detection\n Adjust the alpha value using values between 0 and 1.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                CannyLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                CannyLabelMin.config(text="Minimum:")
                CannyLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                CannyLabelMax.config(text="Maximum")
                CannyEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                CannyEntryMin.config(state="normal")
                CannyEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                CannyEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (CannyLabelMin, CannyLabelMax, CannyEntryMin, CannyEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Canny Edge Detection":
                    try:
                        CannyEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        CannyEntryMax.insert(0, str(SavedWidgetValueList[No][1]))


                    except:
                        print("Error with Shear")

            if Option == "Directional Edge Detection":
                SubDescriptionLabel.config(
                    text="Detect edges in a single direction\n Enter values between 0 and 360.")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                DirEdgeLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                DirEdgeLabelMin.config(text="Minimum:")
                DirEdgeLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                DirEdgeLabelMax.config(text="Maximum:")
                DirEdgeEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                DirEdgeEntryMin.config(state="normal")
                DirEdgeEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                DirEdgeEntryMax.config(state="normal")

                DirEdgeAlphaLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                DirEdgeAlphaLabelMin.config(text="Minimum:")
                DirEdgeAlphaLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                DirEdgeAlphaLabelMax.config(text="Maximum:")
                DirEdgeAlphaEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                DirEdgeAlphaEntryMin.config(state="normal")
                DirEdgeAlphaEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                DirEdgeAlphaEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (DirEdgeAlphaLabelMin, DirEdgeAlphaLabelMax, DirEdgeAlphaEntryMin, DirEdgeAlphaEntryMax,
                     DirEdgeLabelMin, DirEdgeLabelMax, DirEdgeEntryMin, DirEdgeEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Directional Edge Detection":
                    try:
                        DirEdgeEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        DirEdgeEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        DirEdgeAlphaEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        DirEdgeAlphaEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Shear")

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        if SubOptionHistory[Number] != 'Select Edge Detection Options':
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set('Select Edge Detection Options')

        DisplaySubOptions()

    def DisplayColourSegOptions(Number, Frame, LastSub):
        OptionsText.place_forget()
        No = Number
        SegOptions = ["Superpixels", "Voronoi", "Uniform Voronoi", "Regular Grid Voronoi"]
        SubTypeDropdown = ttk.Combobox(Frame, values=SegOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Segmentation Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print("Confirm", No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Superpixels":
                AugList = []

                SupMin = int(SupixEntryMin.get())
                SupMax = int(SupixEntryMax.get())

                AugList.append(iaa.Superpixels(p_replace=(1), n_segments=(SupMin, SupMax)))

                ImportantWidgetValues.extend((SupMin, SupMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Voronoi":
                AugList = []

                VoMinRows = int(VoRowsEntryMin.get())
                VoMaxRows = int(VoRowsEntryMax.get())
                VoMinCols = int(VoColsEntryMin.get())
                VoMaxCols = int(VoColsEntryMax.get())

                points_sampler = iaa.RegularGridPointsSampler(n_rows=random.randint(VoMinRows, VoMaxRows),
                                                              n_cols=random.randint(VoMinCols, VoMaxCols))
                AugList.append(iaa.Voronoi(points_sampler))

                ImportantWidgetValues.extend((VoMinRows, VoMaxRows, VoMinCols, VoMaxCols))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Uniform Voronoi":
                AugList = []

                VoMinRows = int(UniVoRowsEntryMin.get())
                VoMaxRows = int(UniVoRowsEntryMax.get())

                AugList.append(iaa.UniformVoronoi((VoMinRows, VoMaxRows)))

                ImportantWidgetValues.extend((VoMinRows, VoMaxRows))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Regular Grid Voronoi":
                AugList = []

                VoMinRows = int(RegRowsEntryMin.get())
                VoMaxRows = int(RegRowsEntryMax.get())
                VoMinCols = int(RegColsEntryMin.get())
                VoMaxCols = int(RegColsEntryMax.get())

                AugList.append(
                    iaa.RegularGridVoronoi((random.randint(VoMinRows, VoMaxRows), random.randint(VoMinCols, VoMaxCols)),
                                           p_replace=1))

                ImportantWidgetValues.extend((VoMinRows, VoMaxRows, VoMinCols, VoMaxCols))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            try:
                del SubOptionHistory[No]
                # del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())
            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        ########################################### Superpixels Widgets ####################################################

        SupixEntryMin = tk.Entry(Frame, width=6)
        SupixEntryMax = tk.Entry(Frame, width=6)
        SupixLabelMin = tk.Label(Frame, text="", bg="gray86")
        SupixLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Voronoi Widgets ####################################################

        VoRowsEntryMin = tk.Entry(Frame, width=6)
        VoRowsEntryMax = tk.Entry(Frame, width=6)
        VoRowsLabelMin = tk.Label(Frame, text="", bg="gray86")
        VoRowsLabelMax = tk.Label(Frame, text="", bg="gray86")

        VoColsEntryMin = tk.Entry(Frame, width=6)
        VoColsEntryMax = tk.Entry(Frame, width=6)
        VoColsLabelMin = tk.Label(Frame, text="", bg="gray86")
        VoColsLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Uniform Voronoi Widgets ####################################################

        UniVoRowsEntryMin = tk.Entry(Frame, width=6)
        UniVoRowsEntryMax = tk.Entry(Frame, width=6)
        UniVoRowsLabelMin = tk.Label(Frame, text="", bg="gray86")
        UniVoRowsLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### RegularVoronoi Widgets ####################################################

        RegRowsEntryMin = tk.Entry(Frame, width=6)
        RegRowsEntryMax = tk.Entry(Frame, width=6)
        RegRowsLabelMin = tk.Label(Frame, text="", bg="gray86")
        RegRowsLabelMax = tk.Label(Frame, text="", bg="gray86")

        RegColsEntryMin = tk.Entry(Frame, width=6)
        RegColsEntryMax = tk.Entry(Frame, width=6)
        RegColsLabelMin = tk.Label(Frame, text="", bg="gray86")
        RegColsLabelMax = tk.Label(Frame, text="", bg="gray86")

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Superpixels":
                SubDescriptionLabel.config(
                    text="Generate n Superpixels per image.\n Please enter whole positive numbers")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                SupixLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                SupixLabelMin.config(text="Minimum:")
                SupixLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                SupixLabelMax.config(text="Maximum:")
                SupixEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                SupixEntryMin.config(state="normal")
                SupixEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                SupixEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (SupixLabelMin, SupixLabelMax, SupixEntryMin, SupixEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Superpixels":
                    try:
                        SupixEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        SupixEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        pass

            if Option == "Voronoi":
                SubDescriptionLabel.config(
                    text="Averages Colours within a set grid of cells. \n  Enter whole positive numbers")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                VoRowsLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                VoRowsLabelMin.config(text="Rows Min:")
                VoRowsLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                VoRowsLabelMax.config(text="Rows Max:")
                VoRowsEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                VoRowsEntryMin.config(state="normal")
                VoRowsEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                VoRowsEntryMax.config(state="normal")

                VoColsLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                VoColsLabelMin.config(text="Cols Min:")
                VoColsLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                VoColsLabelMax.config(text="Cols Max:")
                VoColsEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                VoColsEntryMin.config(state="normal")
                VoColsEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                VoColsEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (VoColsLabelMin, VoColsLabelMax, VoRowsLabelMin, VoRowsLabelMax, VoRowsEntryMin, VoRowsEntryMax,
                     VoColsEntryMin, VoColsEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Voronoi":
                    try:
                        VoRowsEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        VoRowsEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        VoColsEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        VoColsEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        pass

            if Option == "Uniform Voronoi":
                SubDescriptionLabel.config(
                    text="Averages Colours within a set grouping of cells.\n Values entered are numbers of cells.\n"
                         "Please enter whole values greater than 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                UniVoRowsLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                UniVoRowsLabelMin.config(text="Minimum:")
                UniVoRowsLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                UniVoRowsLabelMax.config(text="Maximum:")
                UniVoRowsEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                UniVoRowsEntryMin.config(state="normal")
                UniVoRowsEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                UniVoRowsEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (UniVoRowsLabelMin, UniVoRowsLabelMax, UniVoRowsEntryMin, UniVoRowsEntryMax,
                     ConfirmButton))
                if SubOptionHistory[No] == "Uniform Voronoi":
                    try:
                        UniVoRowsEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        UniVoRowsEntryMax.insert(0, str(SavedWidgetValueList[No][1]))

                    except:
                        pass

            if Option == "Regular Grid Voronoi":
                SubDescriptionLabel.config(
                    text="Averages Colours within a set grid of cells.\n Drops a random number of cells before averaging\n"
                         "Please enter whole values greater than 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                RegRowsLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                RegRowsLabelMin.config(text="Rows Min:")
                RegRowsLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                RegRowsLabelMax.config(text="Rows Max:")
                RegRowsEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                RegRowsEntryMin.config(state="normal")
                RegRowsEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                RegRowsEntryMax.config(state="normal")

                RegColsLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                RegColsLabelMin.config(text="Cols Min:")
                RegColsLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                RegColsLabelMax.config(text="Cols Max:")
                RegColsEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                RegColsEntryMin.config(state="normal")
                RegColsEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                RegColsEntryMax.config(state="normal")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (RegColsLabelMin, RegColsLabelMax, RegRowsLabelMin, RegRowsLabelMax, RegRowsEntryMin,
                     RegRowsEntryMax,
                     RegColsEntryMin, RegColsEntryMax, ConfirmButton))

                if SubOptionHistory[No] == "Regular Grid Voronoi":
                    try:
                        RegRowsEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        RegRowsEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        RegColsEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        RegColsEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        pass

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        if SubOptionHistory[Number] != 'Select Edge Detection Options':
            try:
                SubTypeDropdown.set(SubOptionHistory[Number])
            except:
                SubTypeDropdown.set('Select Edge Detection Options')

        DisplaySubOptions()

    def DisplayArtSegOptions(Number, Frame, LastSub):
        OptionsText.place_forget()
        No = Number
        ArtOptions = ["Cartoon", "Snowflakes", "Cloudy", "Fog", "Rain"]
        SubTypeDropdown = ttk.Combobox(Frame, values=ArtOptions, width=35, state="readonly",
                                       name=f"dropdown{AugOptions}")
        SubTypeDropdown.bind("<<ComboboxSelected>>", lambda e: DisplaySubOptions())

        SubTypeDropdown.set("Select Artistic Options")
        SubTypeDropdown.place(relx=0.5, rely=0.05, anchor="center")

        def ConfirmOptions():
            print(No)
            ImportantWidgets = []
            ImportantWidgetValues = []

            LabelText = f"{SubTypeDropdown.get()} selected"
            DescriptionList[No].config(text=LabelText)

            try:
                del SubOptionHistory[No]
                del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())

            if SubTypeDropdown.get() == "Cartoon":
                AugList = []

                AugList.append(iaa.Cartoon())

                ImportantWidgetValues.extend((0, 0))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Snowflakes":
                AugList = []

                SizeMin = float(SnowsizeEntryMin.get())
                SizeMax = float(SnowsizeEntryMax.get())
                SpeedMin = float(SnowspeedEntryMin.get())
                SpeedMax = float(SnowspeedEntryMax.get())

                AugList.append(iaa.Snowflakes(flake_size=(SizeMin, SizeMax), speed=(SpeedMin, SpeedMax)))

                ImportantWidgetValues.extend((SizeMin, SizeMax, SpeedMin, SpeedMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Rain":
                AugList = []

                SizeMin = float(RainsizeEntryMin.get())
                SizeMax = float(RainsizeEntryMax.get())
                SpeedMin = float(RainspeedEntryMin.get())
                SpeedMax = float(RainspeedEntryMax.get())

                AugList.append(iaa.Rain(drop_size=(SizeMin, SizeMax), speed=(SpeedMin, SpeedMax)))

                ImportantWidgetValues.extend((SizeMin, SizeMax, SpeedMin, SpeedMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Cloudy":
                AugList = []

                AugList.append(iaa.Clouds())

                ImportantWidgetValues.extend((0, 0))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Fog":
                AugList = []

                AugList.append(iaa.Fog())

                ImportantWidgetValues.extend((0, 0))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            if SubTypeDropdown.get() == "Frost":
                AugList = []

                FrostMin = int(FrostEntryMin.get())
                FrostMax = int(FrostEntryMax.get())

                AugList.append(iaa.imgcorruptlike.Frost(severity=random.randint(FrostMin, FrostMax)))

                ImportantWidgetValues.extend((FrostMin, FrostMax))

                try:

                    del SavedWidgetValueList[No]

                except:
                    pass
                SavedOptionCommand.insert(No, AugList)
                SavedWidgetValueList.insert(No, ImportantWidgetValues)

            try:
                PreviewAugmentations()
            except:
                pass

            try:
                del SubOptionHistory[No]
                # del SavedOptionCommand[No]
            except:
                pass
            SubOptionHistory.insert(No, SubTypeDropdown.get())
            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        ConfirmButton = tk.Button(Frame, text="Confirm", command=ConfirmOptions)

        ########################################### Snowflakes Widgets ####################################################

        SnowsizeEntryMin = tk.Entry(Frame, width=6)
        SnowsizeEntryMax = tk.Entry(Frame, width=6)
        SnowsizeLabelMin = tk.Label(Frame, text="", bg="gray86")
        SnowsizeLabelMax = tk.Label(Frame, text="", bg="gray86")

        SnowspeedEntryMin = tk.Entry(Frame, width=6)
        SnowspeedEntryMax = tk.Entry(Frame, width=6)
        SnowspeedLabelMin = tk.Label(Frame, text="", bg="gray86")
        SnowspeedLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Rain Widgets ####################################################

        RainsizeEntryMin = tk.Entry(Frame, width=6)
        RainsizeEntryMax = tk.Entry(Frame, width=6)
        RainsizeLabelMin = tk.Label(Frame, text="", bg="gray86")
        RainsizeLabelMax = tk.Label(Frame, text="", bg="gray86")

        RainspeedEntryMin = tk.Entry(Frame, width=6)
        RainspeedEntryMax = tk.Entry(Frame, width=6)
        RainspeedLabelMin = tk.Label(Frame, text="", bg="gray86")
        RainspeedLabelMax = tk.Label(Frame, text="", bg="gray86")

        ########################################### Frost Widgets ####################################################

        FrostEntryMin = tk.Entry(Frame, width=6)
        FrostEntryMax = tk.Entry(Frame, width=6)
        FrostLabelMin = tk.Label(Frame, text="", bg="gray86")
        FrostLabelMax = tk.Label(Frame, text="", bg="gray86")

        def DisplaySubOptions():

            SubDescriptionLabel.config(text="")

            CheckWidgetValues = []
            EntryWidgetValues = []

            for x in ActiveWidgetList:
                x.place_forget()
            ActiveWidgetList.clear()
            Option = SubTypeDropdown.get()

            if Option == "Cartoon":
                SubDescriptionLabel.config(
                    text="Creates a cartoon of the image.\n No further options available")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend(
                    (ConfirmButton))

            if Option == "Snowflakes":
                SubDescriptionLabel.config(
                    text="Adds Snowflakes to the images. Enter values between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                SnowsizeLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                SnowsizeLabelMin.config(text="Size Min:")
                SnowsizeLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                SnowsizeLabelMax.config(text="Size Max:")
                SnowsizeEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                SnowsizeEntryMin.config(state="normal")
                SnowsizeEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                SnowsizeEntryMax.config(state="normal")

                SnowspeedLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                SnowspeedLabelMin.config(text="Speed Min:")
                SnowspeedLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                SnowspeedLabelMax.config(text="Speed Max:")
                SnowspeedEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                SnowspeedEntryMin.config(state="normal")
                SnowspeedEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                SnowspeedEntryMax.config(state="normal")

                ActiveWidgetList.extend(
                    (SnowsizeLabelMin, SnowsizeLabelMax, SnowspeedLabelMin, SnowspeedLabelMax, SnowsizeEntryMin,
                     SnowsizeEntryMax,
                     SnowspeedEntryMin, SnowspeedEntryMax, ConfirmButton))

                if SubOptionHistory[No] == "Snowflakes":
                    try:
                        SnowsizeEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        SnowsizeEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        SnowspeedEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        SnowspeedEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Snowflakes")

            if Option == "Cloudy":
                SubDescriptionLabel.config(
                    text="Applies a layer of clouds to the image.\n No further options available")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend((ConfirmButton))

            if Option == "Fog":
                SubDescriptionLabel.config(
                    text="Applies a layer of fog to the image.\n No further options available")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                ActiveWidgetList.extend((ConfirmButton))

            if Option == "Rain":
                SubDescriptionLabel.config(
                    text="Adds Rain to the images. Enter values between 0 and 1")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                RainsizeLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                RainsizeLabelMin.config(text="Size Min:")
                RainsizeLabelMax.place(relx=0.6, rely=0.27, anchor="center")
                RainsizeLabelMax.config(text="Size Max:")
                RainsizeEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                RainsizeEntryMin.config(state="normal")
                RainsizeEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                RainsizeEntryMax.config(state="normal")

                RainspeedLabelMin.place(relx=0.3, rely=0.43, anchor="center")
                RainspeedLabelMin.config(text="Speed Min:")
                RainspeedLabelMax.place(relx=0.6, rely=0.43, anchor="center")
                RainspeedLabelMax.config(text="Speed Max:")
                RainspeedEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                RainspeedEntryMin.config(state="normal")
                RainspeedEntryMax.place(relx=0.75, rely=0.43, anchor="center")
                RainspeedEntryMax.config(state="normal")

                ActiveWidgetList.extend(
                    (RainsizeLabelMin, RainsizeLabelMax, RainspeedLabelMin, RainspeedLabelMax, RainsizeEntryMin,
                     RainsizeEntryMax,
                     RainspeedEntryMin, RainspeedEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Rain":
                    try:
                        RainsizeEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        RainsizeEntryMax.insert(0, str(SavedWidgetValueList[No][1]))
                        RainspeedEntryMin.insert(0, str(SavedWidgetValueList[No][2]))
                        RainspeedEntryMax.insert(0, str(SavedWidgetValueList[No][3]))

                    except:
                        print("Error with Snowflakes")

            if Option == "Frost":
                SubDescriptionLabel.config(
                    text="Adds Frost to the images. Enter whole values between 1 and 5")
                SubDescriptionLabel.place_configure(rely=0.13, relx=0.5, anchor="center")

                ConfirmButton.place(relx=0.5, rely=0.6, anchor="center")

                FrostLabelMin.place(relx=0.3, rely=0.27, anchor="center")
                FrostLabelMin.config(text="Min:")
                FrostLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                FrostLabelMax.config(text="Max:")
                FrostEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                FrostEntryMin.config(state="normal")
                FrostEntryMax.place(relx=0.75, rely=0.27, anchor="center")
                FrostEntryMax.config(state="normal")

                ActiveWidgetList.extend(
                    (FrostLabelMin, FrostLabelMax, FrostEntryMin, FrostEntryMax, ConfirmButton))
                if SubOptionHistory[No] == "Frost":
                    try:
                        FrostEntryMin.insert(0, str(SavedWidgetValueList[No][0]))
                        FrostEntryMax.insert(0, str(SavedWidgetValueList[No][1]))


                    except:
                        print("Error with Snowflakes")

            try:
                del SubOptionHistory[Number]
            except:
                pass
            SubOptionHistory.insert(Number, Option)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        try:
            SubTypeDropdown.set(SubOptionHistory[Number])
        except:
            pass

        DisplaySubOptions()

    inCanvas = False
    def InCanvasOn(e):
        nonlocal inCanvas
        inCanvas = True
      #  print("In")
    def InCanvasOff(e):
        nonlocal inCanvas
        inCanvas = False
       # print("Out")
    def Scrolly(e):  # Scroll through box if necessary
        nonlocal inCanvas
        if inCanvas:
            AugCanvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def DisplayImages():


        LastSetButton.place(anchor="center", relx=0.6, rely=0.67)
        NextSetButton.place(anchor="center", relx=0.7, rely=0.67)
        nonlocal ShownImageLocation
        nonlocal ImageCount
        Step1Entry.delete(0, 10000)
        ImagePaths.clear()
        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        filename2 = filedialog.askdirectory(initialdir=os.getcwd(), title="Please select your top-level folder.")
        Step1Entry.insert(0, filename2)
        for r, d, f in os.walk(Step1Entry.get()):
            for file in f:
                if '.jpg' in file or '.jfif' in file or ".png" in file:
                    print(file)
                    ImagePaths.append(os.path.join(r, file))

        ActiveImages = ImagePaths[:ShownImageLocation]
        ImageCount = len(ImagePaths)
        ImageNoLabel.config(text=f"{ImageCount} images detected")
        for A in ActiveImages:
            ShownImagesPaths.append(A)

        # Make Thumbnail Images
        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            CanvasSize = 200, 200
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)

        CanvasXLoc = 20

        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 900 - TotalWidth

        DeadspaceFraction = round(DeadSpace / 5)
        CanvasXLoc = DeadspaceFraction
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            OrigCanvas.create_image(CanvasXLoc, 125, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

    def NextSet():
        nonlocal ShownImageLocation

        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        PreviewImageThumbnails.clear()
        ShownImageLocation += 4

        if ShownImageLocation >= len(ImagePaths):
            ShownImageLocation = len(ImagePaths)

        First_Image = ShownImageLocation - 4

        if First_Image <= 0:
            First_Image = 0

        for x in ImagePaths[First_Image:ShownImageLocation]:
            ShownImagesPaths.append(x)

        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            print(type(Im))
            CanvasSize = 200, 200
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            print(ThumbnailSize)

            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)



        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 900 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 5)
        CanvasXLoc = DeadspaceFraction
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            print(width, height)
            OrigCanvas.create_image(CanvasXLoc, 125, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

        try:
            PreviewAugmentations()
        except:
            pass

    def PrevSet():
        nonlocal ShownImageLocation

        ShownImagesPaths.clear()
        ShownImagesThumbnails.clear()
        ShownImagesThumbnailsSizes.clear()
        PreviewImageThumbnails.clear()
        ShownImageLocation -= 4

        if ShownImageLocation <= 4:
            ShownImageLocation = 4

        First_Image = ShownImageLocation - 4

        if First_Image <= 0:
            First_Image = 0

        for x in ImagePaths[First_Image:ShownImageLocation]:
            ShownImagesPaths.append(x)

        for Img in ShownImagesPaths:
            Im = Image.open(Img)
            print(type(Im))
            CanvasSize = 200, 200
            Im.thumbnail(CanvasSize, Image.ANTIALIAS)
            ThumbnailSize = Im.size
            ShownImagesThumbnailsSizes.append(ThumbnailSize)
            print(ThumbnailSize)

            PreviewImage = ImageTk.PhotoImage(Im)
            ShownImagesThumbnails.append(PreviewImage)



        TotalWidth = 0
        for I in ShownImagesThumbnailsSizes:
            width = I[0]
            TotalWidth += width

        DeadSpace = 900 - TotalWidth
        DeadspaceFraction = round(DeadSpace / 5)
        CanvasXLoc = DeadspaceFraction
        for I in range(len(ShownImagesThumbnails)):
            Nigel = ShownImagesThumbnails[I]
            width, height = ShownImagesThumbnailsSizes[I]
            print(width, height)
            OrigCanvas.create_image(CanvasXLoc, 125, image=Nigel, anchor="w")
            CanvasXLoc += (width + DeadspaceFraction)

        try:
            PreviewAugmentations()
        except:
            pass

    def ImageNumberPlacement(event):
        if OptionDropdown.get() == "Apply To Specific Number Of Images":
            NoImageLabel.place(anchor="center", relx=0.65, rely=0.76)
            NoImageEntry.place(anchor="center", relx=0.71, rely=0.76)
        else:
            try:
                NoImageEntry.place_forget()
                NoImageLabel.place_forget()
            except:
                pass

    def NewSetPlacement():
        if SetVar.get() == 2:

            NewSetLabel2.place(anchor="center", relx=0.665, rely=0.83)
            NewSetEntry.place(anchor="center", relx=0.655, rely=0.86)
            NewSetBrowse.place(anchor="center", relx=0.765, rely=0.86)

        elif SetVar.get() == 1:

            NewSetLabel2.place_forget()
            NewSetEntry.place_forget()
            NewSetBrowse.place_forget()

        elif SetVar.get() == 3:

            NewSetLabel2.place_forget()
            NewSetEntry.place_forget()
            NewSetBrowse.place_forget()

    def GetNewSetLoc():
        NewSetEntry.delete(0, 1000)
        NewSetFolder = filedialog.askdirectory(initialdir=os.getcwd(),
                                               title="Please select your output folder.")
        NewSetEntry.insert(0, NewSetFolder)

    def FinalPreChecks():
        Error = 0
        Step1Entry.config(fg="black", text="Step 1: Please select the images you want to augment:")
        AugInstructText.config(fg="black")
        HowLabel.config(fg="black")
        NewSetLabel2.config(fg="black", text="Select your output folder", bg="gray83")

        if len(Step1Entry.get()) == 0:
            Step1Text.config(fg="red")
            Error += 1

        if not os.path.isdir(str(Step1Entry.get())):
            Step1Text.config(fg="red", text="Not a valid folder")
            Error += 1

        if len(SavedOptionCommand[0]) == 0:
            AugInstructText.config(fg="red")
            Error += 1
        if len(OptionDropdown.get()) == 0:
            HowLabel.config(fg="red")
            Error += 1
        if SetVar.get() == 2:
            if len(NewSetEntry.get()) == 0:
                NewSetLabel2.config(fg="red")
                Error += 1

            elif not os.path.isdir(str(NewSetEntry.get())):
                NewSetLabel2.config(fg="red", text="Not a valid folder")
                Error += 1

        return Error

    def Aug_Thread_Threading():
        nonlocal Processed_Images
        nonlocal ImagesForAug
        Errors = FinalPreChecks()

        if Errors == 0:
            if SetVar.get() == 2:
                RecreateFolderStructure()

            if OptionDropdown.get() == "Apply To Random Amount Of Images":
                ImageNumber = random.randint(1, len(ImagePaths))
                SelectedImages = random.sample(ImagePaths, ImageNumber)
                ImagesForAug = SelectedImages
            if OptionDropdown.get() == "Apply To Specific Number Of Images":
                ImageNumber = int(NoImageEntry.get())
                SelectedImages = random.sample(ImagePaths, ImageNumber)
                ImagesForAug = SelectedImages
            if OptionDropdown.get() == "Apply To All Images":
                ImagesForAug = ImagePaths

            t = threading.Thread(target=Aug_Threading)
            t.start()
            ProgressPopup(len(ImagePaths))

    def Aug_Threading():
        if len(ImagePaths) > 0:
            pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
            for I in range(len(ImagesForAug)):
                # print(ImagePaths[I])
                pool.submit(AugCore, ImagesForAug[I])
            pool.shutdown(wait=True)

    def RecreateFolderStructure():
        OriginalTopFolder = Step1Entry.get()
        NewSetTopFolder = NewSetEntry.get()

        for In in ImagePaths:
            OriginalImageName = (os.path.split(In)[1]).split(".")[0]
            OriginalImageLocation = os.path.split(In)[0]
            OriginalImageShortLocation = OriginalImageLocation.replace(OriginalTopFolder, "")
            # os.path.join kept fucking up here, no idea why.
            NewFolder = str(NewSetTopFolder) + str(OriginalImageShortLocation) + "\\" + str(OriginalImageName)

            try:
                os.makedirs(NewFolder)
            except Exception as e:
                pass

    def AugCore(ImagePath):
        global Images_Processed
        Flat_Aug = []
        Orig_Image = io.imread(ImagePath)

        for x in SavedOptionCommand:
            if type(x) is list:
                for item in x:
                    Flat_Aug.append(item)
            else:
                Flat_Aug.append(x)

        if len(Flat_Aug) > 1:
            Augmentation = iaa.Sequential(Flat_Aug)
        else:
            Augmentation = Flat_Aug[0]

        # print(Augmentation)

        Image_Out = Augmentation.augment_image(Orig_Image)
        # print(Image_Out)
        # print(SetVar.get())

        if SetVar.get() == 2:

            Chop = ImagePath.replace(Step1Entry.get(), "")

            NewLoc = str(NewSetEntry.get()) + Chop
            print(NewLoc)
            io.imsave(NewLoc, Image_Out)
            Images_Processed += 1

        elif SetVar.get() == 1:
            io.imsave(ImagePath, Image_Out)
            Images_Processed += 1

        elif SetVar.get() == 3:
            # print("Adding to existing set")
            i = 1
            Chop = os.path.split(ImagePath)[-1]
            ChoppedPath = os.path.split(ImagePath)[0]
            ImageName = os.path.splitext(Chop)[0]
            Extention = os.path.splitext(Chop)[1]
            NewImgName = ImageName + f"Augmented ({i})" + Extention
            NewImagePath = os.path.join(ChoppedPath, NewImgName)

            while os.path.exists(NewImagePath):
                i += 1
                print("Exists, iteraing")
                NewImgName = ImageName + f"Augmented ({i})" + Extention
                NewImagePath = os.path.join(ChoppedPath, NewImgName)

            io.imsave(NewImagePath, Image_Out)
            Images_Processed += 1

        print(Processed_Images)



    # Aug Interface Main Screen
    LargeFont = font.Font(family="Helvetica", size=15)
    LargeSkinnyFont = font.Font(size=13)



    Step1Text = tk.Label(MainFrame, fg="black", text="Please select the images you want to augment:",
                         font=20, bg="gray83")
    Step1Text.place(anchor="center", relx=0.175, rely=0.02)
    Step1Entry = tk.Entry(MainFrame, width=30, font=Large_Font)
    Step1Entry.place(anchor="center", relx=0.16, rely=0.05)
    ButtonForFiles = tk.Button(MainFrame, text="Browse...", command=DisplayImages, font=Small_Font)
    ButtonForFiles.place(anchor="center", relx=0.3, rely=0.05)

    ImageNoLabel = tk.Label(MainFrame, text="", font=20, fg="blue", bg="gray83")
    ImageNoLabel.place(anchor="center", relx=0.175, rely=0.08)

    AugInstructText = tk.Label(MainFrame, text="Select your augmentations", font=20, bg="gray83")
    AugInstructText.place(anchor="center", relx=0.175, rely=0.11)

    AugFrameOutter = tk.Frame(MainFrame, width=420, height=300, borderwidth=2, relief="ridge")
    AugFrameOutter.place(anchor="center", relx=0.18, rely=0.30)
    AugFrameOutter.bind("<Enter>", InCanvasOn)
    AugFrameOutter.bind("<Leave>", InCanvasOff)

    AugCanvas = tk.Canvas(AugFrameOutter, width=420, height=300, bg="gray86", scrollregion=(0, 0, 500, 300),
                          highlightthickness=0)
    AugScroll = tk.Scrollbar(AugFrameOutter, orient="vertical", command=AugCanvas.yview)
    AugScroll.pack(side="left", fill="y")
    AugCanvas.config(yscrollcommand=AugScroll.set)
    AugCanvas.pack(side="left", expand=True, fill="both")
    AugCanvas.bind_all("<MouseWheel>", Scrolly)

    AddOptionButton = tk.Button(MainFrame, text="Add Augmentation", font=10, width=20, command=AddOption)
    AddOptionButton.place(relx=0.179, rely=0.5,anchor="center")

    OptionsFrame = tk.Frame(MainFrame, width=440, height=400, borderwidth=2, relief="ridge", bg="grey86")
    OptionsFrame.place(anchor="center", relx=0.179, rely=0.75)

    SubDescriptionLabel = tk.Label(OptionsFrame, text="", bg="gray86")

    OptionsText = tk.Label(OptionsFrame, text="Image Options Appear Here.", bg="grey86")
    OptionsText.place(anchor="center", relx=0.5, rely=0.5)

    OrigLabel = tk.Label(MainFrame, text="Original Images", bg="gray83", font=Large_Font)

    OrigLabel.place(anchor="center", relx=0.665, rely=0.012)



    OrigCanvas = tk.Canvas(MainFrame, width=900, height=250)
    OrigCanvas.place(anchor="center", relx=0.665, rely=0.17)

    AugLabel = tk.Label(MainFrame, text="Augmented Images", bg="gray83", font=Large_Font)
    AugLabel.place(anchor="center", relx=0.665, rely=0.33)

    PrevCanvas = tk.Canvas(MainFrame, width=900, height=250)
    PrevCanvas.place(anchor="center", relx=0.665, rely=0.50)

    LastSetButton = tk.Button(MainFrame, text="Prev", width=10, command=PrevSet)

    NextSetButton = tk.Button(MainFrame, text="Next", width=10, command=NextSet)

    HowLabel = tk.Label(MainFrame, text="How do you want these changes applied?", font=20, bg="gray83")
    HowLabel.place(anchor="center", relx=0.665, rely=0.7)

    OptionList2 = ["Apply To All Images", "Apply To Random Amount Of Images", "Apply To Specific Number Of Images"]
    OptionDropdown = ttk.Combobox(MainFrame, values=OptionList2, width=35, state="readonly")
    OptionDropdown.place(relx=0.665, rely=0.73, anchor="center")
    OptionDropdown.bind("<<ComboboxSelected>>", ImageNumberPlacement)

    NoImageLabel = tk.Label(MainFrame, text="Number of Images", bg="gray83")
    NoImageEntry = tk.Entry(MainFrame, width=7)

    SetVar = tk.IntVar()
    SetVar.set(1)
    SameSetRadio = tk.Radiobutton(MainFrame, text="Replace Selected Images", variable=SetVar, value=1,
                                  command=NewSetPlacement, bg="gray83")
    SameSetRadio.place(anchor="center", relx=0.56, rely=0.8)
    NewSetRadio = tk.Radiobutton(MainFrame, text="Create New Image Set", variable=SetVar, value=2,
                                 command=NewSetPlacement, bg="gray83")
    NewSetRadio.place(anchor="center", relx=0.665, rely=0.8)

    AddSetRadio = tk.Radiobutton(MainFrame, text="Add to Selected Images", variable=SetVar, value=3,
                                 command=NewSetPlacement, bg="gray83")
    AddSetRadio.place(anchor="center", relx=0.77, rely=0.8)

    NewSetLabel2 = tk.Label(MainFrame, text="Select your output folder", bg="gray83")
    NewSetEntry = tk.Entry(MainFrame, width=40)
    NewSetBrowse = tk.Button(MainFrame, text="Browse...", command=GetNewSetLoc)


    BeginButton = tk.Button(MainFrame, text="Start", width=10, command=Aug_Thread_Threading)
    BeginButton.place(anchor="center", relx=0.665, rely=0.9)



    ################################# Options Widets ###############################################

    DescLabel = tk.Label(OptionsFrame, text="", bg="grey86")
    DescLabel.place(relx=0.5, rely=0.05, anchor="center")
    #################################################################################################

    AugFrameInner1 = tk.Frame(AugCanvas, height=100, width=380, relief="raised", borderwidth=1, bg="gray88")
    InnerFramesList.append(AugFrameInner1)
    InnerFramesIDs.append(ActiveAugmentations)

    AugLabel1 = tk.Label(AugFrameInner1, text=f"Augmentation {1}", name=f"label{AugOptions}",bg="gray88")
    AugLabel1.place(relx=0.5, rely=0.2, anchor="center")
    LabelList.append(AugLabel1)

    DescriptionLabel1 = tk.Label(AugFrameInner1, text="", name=f"desclabel{0}", bg="gray88")
    DescriptionLabel1.place(relx=0.5, rely=0.8, anchor="center")
    DescriptionList.append(DescriptionLabel1)

    OptionButton1 = tk.Button(AugFrameInner1, text="Open", name=f"options{0}",command=lambda: OpenOption(str(OptionButton1)), bg="gray88", width = 6)
    OptionButton1.place(relx=0.85, rely=0.3, anchor="center")
    OPButtList.append(str(OptionButton1))

    RemoveButton1 = tk.Button(AugFrameInner1, text="Remove", name=f"remove{0}", bg="gray88", command=lambda: RemoveOption(RemoveButton1))
    RemoveButton1.place(relx=0.85, rely=0.7, anchor="center")
    RemButtList.append(str(RemoveButton1))

    OptionList1 = ["Colour Options","Size Transformations", "Brightness and Contrast", "Sharpen and Emboss", "Noise", "Dropout", "Blur",
                  "Geometric Transformations", "Edge Detection", "Colour Segmentation", "Artistic Options"]
    TypeDropdown1 = ttk.Combobox(AugFrameInner1, values=OptionList1, width=35, state="readonly",
                                name=f"dropdown{AugOptions}")

    OptionHistory.append(" ")
    SubOptionHistory.append(" ")
    SavedOptionsList.append(" ")
    SavedOptionCommand.append([])
    SavedWidgetValueList.append([])

    TypeDropdown1.set("Select Augmentation Type")
    TypeDropdown1.place(relx=0.4, rely=0.5, anchor="center")
    TypeDropDownList.append(TypeDropdown1)
    TypeDropdown1.bind("<<ComboboxSelected>>", lambda e: OpenOption(str(OptionButton1)))
    TypeDropdown1.unbind_class("TCombobox", "<MouseWheel>")

    ID1 = AugCanvas.create_window(210, 0, window=AugFrameInner1, anchor="n")
    OptionWidgetListID.append(ID1)


print("Loaded")
tk.mainloop()
