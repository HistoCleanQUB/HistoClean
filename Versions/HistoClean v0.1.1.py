# Import Required Packages
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

# Home Screen and General Processes
print("Loading...")


SplashRoot = tk.Tk()
ScreenWidthMiddle = int(SplashRoot.winfo_screenwidth() / 2)
ScreenHeightMiddle = int(SplashRoot.winfo_screenheight() / 2)

canvas = tk.Canvas(SplashRoot, width=700, height=600, bg='white', highlightthickness=0)
canvas.place(relx=0.5, rely=0.5, anchor="center")
canvas.master.overrideredirect(True)
canvas.master.geometry(f"700x579+{ScreenWidthMiddle - 350}+{ScreenHeightMiddle - 280}")
canvas.master.wm_attributes("-transparentcolor","white")
canvas.master.wm_attributes("-alpha",1)
canvas.master.wm_attributes("-topmost",True)
canvas.master.lift()
img = ImageTk.PhotoImage(Image.open("Icon/HistoSquare.png"))
canvas.create_image(0, 0, anchor="nw", image=img)





def mainroot():

    # For selecting which module to go to next
    def Select_Process():
        if Var.get() == 0:
            HomeScreen.withdraw()
            Patch_App()
        if Var.get() == 1:
            HomeScreen.withdraw()
            White_App()
        elif Var.get() == 2:
            HomeScreen.withdraw()
            Balance_App()
        elif Var.get() == 3:
            HomeScreen.withdraw()
            Normalisation_App()
        elif Var.get() == 4:
            HomeScreen.withdraw()
            Augmentation_App()


    # Universal function for retuning to hoe screen when module tab is closed
    def Close_And_Home(ActiveWindow):
        HomeScreen.deiconify()
        ActiveWindow.destroy()


    # Function for limiting entry widgets to integers
    def IntOnly(Char, acttyp):
        if acttyp == "1":
            if not Char.isdigit():
                return False
        return True


    # Calculate number of CPU cores and set this to the number of threads available for multithreading
    ThreadsAllowed = int(multiprocessing.cpu_count())

    HomeScreen = tk.Tk()

    # Get size of users screen
    ScreenWidthMiddle = int(HomeScreen.winfo_screenwidth() / 2)
    ScreenHeightMiddle = int(HomeScreen.winfo_screenheight() / 2)

    HomeScreen.title("HistoClean")
    HomeScreen.geometry(f"800x300+{ScreenWidthMiddle - 400}+{ScreenHeightMiddle - 150}")
    HomeScreen.resizable(False, False)
    HomeScreen.iconbitmap(r"Icon/HistoSquare.ico")

    # Labels for the Introductory text
    IntroText = tk.Label(HomeScreen, fg="black", text=r"Welcome to HistoClean!", font=90, anchor="center")
    IntroText.place(anchor="center", relx=0.5, rely=0.05)
    IntroTextTwo = tk.Label(HomeScreen, text="Please select your desired pre-processing method", font=90, anchor="center")
    IntroTextTwo.place(anchor="center", relx=0.5, rely=0.15)

    # Help Description Box
    HelpLabel = tk.Label(HomeScreen, text="")
    HelpLabel.pack(side="bottom", fill="x")

    # Radio buttons for Initial options
    Var = tk.IntVar()
    Tile_Checker = tk.Radiobutton(HomeScreen, variable=Var, value=0, text="Image Tiling", font=20)
    Tile_Checker.place(anchor="center", relx=0.5, rely=0.27)
    Patch_Description = "Subsample, patch and resize your images using this module"
    Tile_Checker.bind("<Enter>", lambda e: HelpLabel.config(text=Patch_Description))
    Tile_Checker.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    WS_Var_Checker = tk.Radiobutton(HomeScreen, variable=Var, value=1, text="Whitespace Filter", font=20)
    WS_Var_Checker.place(anchor="center", relx=0.5, rely=0.37)
    WSDescription = "Detect whitespace in images and remove according to your own threshold"
    WS_Var_Checker.bind("<Enter>", lambda e: HelpLabel.config(text=WSDescription))
    WS_Var_Checker.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    Balance_Var_Checker = tk.Radiobutton(HomeScreen, variable=Var, value=2, text="Quick Balance Dataset", font=20)
    Balance_Var_Checker.place(anchor="center", relx=0.5, rely=0.47)
    Balance_Description = "Balance the number of images in two seperate folders using random flips and rotations"
    Balance_Var_Checker.bind("<Enter>", lambda e: HelpLabel.config(text=Balance_Description))
    Balance_Var_Checker.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    Norm_Var_Checker = tk.Radiobutton(HomeScreen, variable=Var, value=3, text="Image Normalisation", font=20)
    Norm_Var_Checker.place(anchor="center", relx=0.5, rely=0.57)
    Norm_Description = "Normalise th colour range of your images to a single image"
    Norm_Var_Checker.bind("<Enter>", lambda e: HelpLabel.config(text=Norm_Description))
    Norm_Var_Checker.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    Aug_Var_Checker = tk.Radiobutton(HomeScreen, variable=Var, value=4, text="Image Augmentation", font=20)
    Aug_Var_Checker.place(anchor="center", relx=0.5, rely=0.67)
    Aug_Description = "Augment your images to expand your dataset or accentuate image features"
    Aug_Var_Checker.bind("<Enter>", lambda e: HelpLabel.config(text=Aug_Description))
    Aug_Var_Checker.bind("<Leave>", lambda e: HelpLabel.config(text=""))

    Exit_Button = tk.Button(HomeScreen, text="Exit", height=1, width=10)
    Exit_Button.place(anchor="center", relx=0.4, rely=0.85)
    Next_Button = tk.Button(HomeScreen, text="Next", height=1, width=10, command=Select_Process)
    Next_Button.place(anchor="center", relx=0.6, rely=0.85)


    ################################################# Image Patching Application  #########################################

    def Patch_App():
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

        ListList = [InputImages, OriginalWidthList, OriginalHeightList, ImageLocationXList, ImageLocationYList,
                    DownfactorList,
                    ThumbnailImages, ThumbnailWidthList, ThumbnailHeightList, DrawnLines]
        CurrentImage = 0
        CanvasWidth = 650
        CanvasHeight = 300
        Images_Processed = 0
        Images_Loaded = 0

        Patch_Interface = tk.Toplevel(HomeScreen)
        Patch_Interface.geometry(f"900x900+{ScreenWidthMiddle - 450}+{ScreenHeightMiddle - 450}")
        Patch_Interface.wm_title("Image Tiling")
        Patch_Interface.resizable(True, True)
        Patch_Interface.protocol("WM_DELETE_WINDOW", lambda: Close_And_Home(Patch_Interface))
        Patch_Interface.iconbitmap(r"Icon/HistoSquare.ico")

        # Get the images and make into thumbnails
        def LoadImages():
            nonlocal CurrentImage
            # Reset all inputs
            CurrentImage = 0
            PreviewCanvas.delete(tk.ALL)
            Step1Entry.delete(0, 10000)
            for li in ListList:
                li.clear()
            filename2 = filedialog.askdirectory(initialdir=os.getcwd(), title="Please select your top-level folder.")
            Step1Entry.insert(0, filename2)
            for r, d, f in os.walk(Step1Entry.get()):
                for file in f:
                    extension = file.split(".")
                    extension = extension[-1]
                    print(extension)

                    if 'jpg' in extension or 'jfif' in extension or "png" in extension or 'JPG' in extension:
                        print(file)
                        InputImages.append(os.path.join(r, file))
            ImageDetectionLabel.config(text=f"{len(InputImages)} Images Detected")
            PreviewCanvas.create_text(325, 150, text="Loading Images...", font=font.Font(size=40, slant="italic"))

            CanvasSize = CanvasWidth, CanvasHeight

            def ProcessImage(Im):
                nonlocal Images_Loaded
                # print(Im)
                ActiveImage = Image.open(Im)
                OrigWidth, OrigHeight = ActiveImage.size
                ActiveImage.thumbnail(CanvasSize, Image.ANTIALIAS)
                ReworkedWidth, ReworkedHeight = ActiveImage.size
                ImageTopBucketX = (CanvasWidth / 2) - (ReworkedWidth / 2)
                ImageTopBucketY = (CanvasHeight / 2) - (ReworkedHeight / 2)
                Downfactor = ReworkedWidth / OrigWidth
                PreviewImage = ImageTk.PhotoImage(ActiveImage)
                ReworkedWidth, ReworkedHeight = ActiveImage.size
                Images_Loaded += 1
                return OrigWidth, OrigHeight, ImageTopBucketX, ImageTopBucketY, Downfactor, PreviewImage, ReworkedWidth, ReworkedHeight

            def PreviewThread():

                pool = ThreadPoolExecutor(max_workers=round(int(ThreadsAllowed) * 1.5))
                for x in range(len(InputImages)):
                    future = pool.submit(ProcessImage, InputImages[x])
                    OrigWidth, OrigHeight, ImageTopBucketX, ImageTopBucketY, DownFactor, PreviewImage, ReworkedWidth, ReworkedHeight = future.result()
                    OriginalWidthList.append(OrigWidth)
                    OriginalHeightList.append(OrigHeight)
                    ImageLocationXList.append(ImageTopBucketX)
                    ImageLocationYList.append(ImageTopBucketY)
                    DownfactorList.append(DownFactor)
                    ThumbnailImages.append(PreviewImage)
                    ThumbnailWidthList.append(ReworkedWidth)
                    ThumbnailHeightList.append(ReworkedHeight)

                pool.shutdown(wait=True)
                PreviewCanvas.delete(tk.ALL)
                PreviewCanvas.create_image(CanvasWidth / 2, CanvasHeight / 2, image=ThumbnailImages[0], anchor="center")
                ImageSizeLabel.config(text=f"Image Size: {OriginalWidthList[0]} X {OriginalHeightList[0]} pixels")

            threading.Thread(target=PreviewThread).start()
            pgbarImg.place(anchor="center", relx=0.5, rely=0.55)
            Update_Progress_Img()

        # Draw the lines showing where the image is split up
        def PreviewGrid():

            try:
                for x in DrawnLines:
                    PreviewCanvas.delete(x)
            except:
                pass

            DownFactor = DownfactorList[CurrentImage]
            WidthAdjusted = int(WidthEntry.get()) * DownFactor
            HeightAdjusted = int(WidthEntry.get()) * DownFactor
            XRectanglesNeeded = math.floor(ThumbnailWidthList[CurrentImage] / WidthAdjusted)
            YRectanglesNeeded = math.floor(ThumbnailHeightList[CurrentImage] / HeightAdjusted)

            for x in range(0, XRectanglesNeeded, 1):
                TopLeftX = ImageLocationXList[CurrentImage]
                TopLeftX = TopLeftX + (WidthAdjusted * x)
                BottomRightX = TopLeftX + WidthAdjusted

                for y in range(0, YRectanglesNeeded, 1):
                    TopLeftY = ImageLocationYList[CurrentImage]
                    TopLeftY = TopLeftY + (HeightAdjusted * y)
                    BottomRightY = TopLeftY + HeightAdjusted

                    Rect = PreviewCanvas.create_rectangle(int(TopLeftX), int(TopLeftY), int(BottomRightX),
                                                          int(BottomRightY), width=2, outline="red")
                    DrawnLines.append(Rect)

            if XRectanglesNeeded >= 1 and YRectanglesNeeded >= 1:
                Line = PreviewCanvas.create_line(int(ImageLocationXList[CurrentImage]),
                                                 int(ImageLocationYList[CurrentImage] + 2), int(BottomRightX),
                                                 int(ImageLocationYList[CurrentImage] + 2), width=4, fill="red")
                DrawnLines.append(Line)
                ImageSizeLabel.config(fg="blue",
                                      text=f"Image Size: {OriginalWidthList[CurrentImage]} X {OriginalHeightList[CurrentImage]} pixels")
            else:
                ImageSizeLabel.config(fg="red",
                                      text=f"Image Size: {OriginalWidthList[CurrentImage]} X {OriginalHeightList[CurrentImage]} pixels. Image Too Small to produce tiles of this size")

        # Move to next Image
        def NextImage():
            nonlocal CurrentImage
            if CurrentImage < len(InputImages) - 1:
                CurrentImage += 1
            PreviewCanvas.delete(tk.ALL)
            PreviewCanvas.create_image(CanvasWidth / 2, CanvasHeight / 2, image=ThumbnailImages[CurrentImage],
                                       anchor="center")
            ImageSizeLabel.config(fg="blue",
                                  text=f"Image Size: {OriginalWidthList[CurrentImage]} X {OriginalHeightList[CurrentImage]} pixels")
            PreviewGrid()

        # Move to previous Image
        def PrevImage():
            nonlocal CurrentImage
            if CurrentImage > 0:
                CurrentImage -= 1
            PreviewCanvas.delete(tk.ALL)
            PreviewCanvas.create_image(CanvasWidth / 2, CanvasHeight / 2, image=ThumbnailImages[CurrentImage],
                                       anchor="center")
            ImageSizeLabel.config(fg="blue",
                                  text=f"Image Size: {OriginalWidthList[CurrentImage]} X {OriginalHeightList[CurrentImage]} pixels")
            PreviewGrid()

        # Get the save folder location
        def GetSaveFolder():
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
            Step1Text.config(fg="black")
            SizeLabel.config(fg="black")
            SaveLabel.config(fg="black")

            if len(Step1Entry.get()) < 1:
                Step1Text.config(fg="red")
            if len(WidthEntry.get()) < 1:
                SizeLabel.config(fg="red")
            if len(SaveEntry.get()) < 1:
                SaveLabel.config(fg="red")
            else:
                PatchThreadThreading()

        # Make folder for each image
        def RecreateFolderStructure():
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

        def Update_Progress_Img():
            nonlocal Images_Loaded
            nonlocal InputImages
            while Images_Loaded < len(InputImages):
                print(f"{Images_Loaded}/{len(InputImages)}")
                pgbarImg.config(maximum=len(InputImages))
                StartButton.config(state="disabled")
                ProgressVarImg.set(Images_Loaded)
                # Progresslabel.config(text=f"Progress: {Images_Loaded} / {len(InputImages)} complete")
                Patch_Interface.update()
                time.sleep(0.000001)
            pgbarImg.place_forget()
            StartButton.config(state="normal")

        # Patches up images and saves them
        def PatchCore(Ima):
            nonlocal Images_Processed
            OriginalTopFolder = Step1Entry.get()
            NewSetTopFolder = SaveEntry.get()
            ImageName = os.path.split(Ima)[-1]
            ImageExtention = "." + str(ImageName.split(".")[-1])
            OriginalImageLocation = os.path.split(Ima)[0]
            OriginalLocation_NoTopFolder = OriginalImageLocation.replace(OriginalTopFolder, "")
            ImageName_NoEXT = ImageName.split(".")[0]
            NewImageSaveLocation = str(NewSetTopFolder) + str(OriginalLocation_NoTopFolder) + "\\" + str(ImageName_NoEXT)

            Im = Image.open(Ima)
            ImageArray = asarray(Im)
            if len(ImageArray.shape) == 3:
                Dim = (int(WidthEntry.get()), int(WidthEntry.get()), 3)
            else:
                Dim = (int(WidthEntry.get()), int(WidthEntry.get()))

            if ImageArray.shape[0] < int(WidthEntry.get()) or ImageArray.shape[1] < int(WidthEntry.get()):
                SmallList.append(ImageName)
                Images_Processed += 1

            Patches = patchify(ImageArray, Dim, step=int(WidthEntry.get()))

            if ExtentionVar.get() == 0:
                Extension = ".jpg"

            elif ExtentionVar.get() == 1:
                Extension = ".png"

            elif ExtentionVar.get() == 2:
                Extension = ImageExtention
            if len(ImageArray.shape) == 3:
                for x in range(Patches.shape[0]):
                    for y in range(Patches.shape[1]):
                        Patch = Patches[x][y][0]
                        PatchSaveFile = os.path.join(NewImageSaveLocation, f"Patch {x},{y}{Extension}")
                        Image.fromarray(Patch).save(PatchSaveFile)
            else:
                for x in range(Patches.shape[0]):
                    for y in range(Patches.shape[1]):
                        Patch = Patches[x][y]
                        PatchSaveFile = os.path.join(NewImageSaveLocation, f"Patch {x},{y}{Extension}")
                        Image.fromarray(Patch).save(PatchSaveFile)
            Images_Processed += 1

        def PatchThreadThreading():
            nonlocal Images_Processed
            Images_Processed = 0
            t = threading.Thread(target=PatchThread)
            t.start()
            RecreateFolderStructure()
            Update_Progress()

        def PatchThread():
            print(len(InputImages))
            pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
            for x in range(len(InputImages)):
                pool.submit(PatchCore, InputImages[x])
            pool.shutdown(wait=True)
            print(SmallList)

        def Update_Progress():
            nonlocal Images_Processed
            nonlocal InputImages
            while Images_Processed < len(InputImages):
                print(f"{Images_Processed}/{len(InputImages)}")
                pgbar.config(maximum=len(InputImages))
                StartButton.config(state="disabled")
                ProgressVarCol.set(Images_Processed)
                Progresslabel.config(text=f"Progress: {Images_Processed} / {len(InputImages)} complete")
                Patch_Interface.update()
                time.sleep(0.000001)

            Progresslabel.config(text=f"Progress {Images_Processed} / {len(InputImages)} complete")
            ProgressVarCol.set(Images_Processed)
            Patch_Interface.update()
            StartButton.config(state="normal")
            Pop()

        def Pop():

            Popup = tk.Toplevel(HomeScreen)
            Popup.geometry(f"200x100+{ScreenWidthMiddle - 100}+{ScreenHeightMiddle - 50}")
            Popup.title("Complete")
            Popup.resizable(False, False)
            Popup.iconbitmap(r"Icon/HistoSquare.ico")
            Info = tk.Label(Popup, text="Process Complete\n"
                                        "Stay here or return to home screen?")

            Popup.grab_set()

            def ReturnHome():
                Popup.destroy()
                HomeScreen.deiconify()
                Patch_Interface.destroy()

            def Stay():
                Popup.destroy()
                Popup.grab_release()
                ResetAll()

            Info.place(relx=0.5, rely=0.2, anchor="center")
            Home_Button = tk.Button(Popup, text="Home", command=ReturnHome, width=10)
            Home_Button.place(relx=0.25, rely=0.6, anchor="center")
            Stay_Button = tk.Button(Popup, text="Stay", command=Stay, width=10)
            Stay_Button.place(relx=0.75, rely=0.6, anchor="center")

        def ResetAll():

            nonlocal Images_Processed
            nonlocal ListList
            nonlocal CurrentImage
            PreviewCanvas.delete(tk.ALL)
            Step1Entry.delete(0, 10000)
            WidthEntry.delete(0, 10000)
            SaveEntry.delete(0, 10000)
            CurrentImage = 0

            InputImages.clear()
            OriginalWidthList.clear()
            OriginalHeightList.clear()
            ImageLocationXList.clear()
            ImageLocationYList.clear()
            DownfactorList.clear()
            ThumbnailImages.clear()
            ThumbnailWidthList.clear()
            ThumbnailHeightList.clear()
            DrawnLines.clear()
            SmallList.clear()

            ListList = [InputImages, OriginalWidthList, OriginalHeightList, ImageLocationXList, ImageLocationYList,
                        DownfactorList,
                        ThumbnailImages, ThumbnailWidthList, ThumbnailHeightList, DrawnLines]

            ProgressVarCol.set(0)
            Progresslabel.config(text="")
            ImageDetectionLabel.config(text="")
            ImageSizeLabel.config(text="")

            Images_Processed = 0

        ################################### Patch Interface Widgets ######################################################
        InfoLabel = tk.Label(Patch_Interface, text=f"This module will allow you to tile up your images")
        InfoLabel.place(anchor="center", relx=0.5, rely=0.02)

        Step1Text = tk.Label(Patch_Interface, fg="black", text="Select your image folder:")
        Step1Text.place(anchor="center", relx=0.5, rely=0.06)

        Step1Entry = tk.Entry(Patch_Interface, width=40, justify="center")
        Step1Entry.place(anchor="center", relx=0.475, rely=0.1)

        ButtonForFiles = tk.Button(Patch_Interface, text="Browse...", command=LoadImages)
        ButtonForFiles.place(anchor="center", relx=0.65, rely=0.1)

        ImageDetectionLabel = tk.Label(Patch_Interface, fg="blue", text="")
        ImageDetectionLabel.place(anchor="center", relx=0.5, rely=0.13)

        PreviewCanvas = tk.Canvas(Patch_Interface, height=300, width=650)
        PreviewCanvas.place(anchor="n", relx=0.5, rely=0.17)

        ImageSizeLabel = tk.Label(Patch_Interface, text="", fg="blue")
        ImageSizeLabel.place(anchor="center", relx=0.5, rely=0.57)

        PrevButton = tk.Button(Patch_Interface, text="Previous", command=PrevImage)
        PrevButton.place(anchor="center", relx=0.1, rely=0.4)

        NextButton = tk.Button(Patch_Interface, text="Next", command=NextImage)
        NextButton.place(anchor="center", relx=0.9, rely=0.4)

        SizeLabel = tk.Label(Patch_Interface, text="Please enter your tile size (pixels)")
        SizeLabel.place(anchor="center", relx=0.5, rely=0.6)

        Validation = (Patch_Interface.register(IntOnly))

        WidthEntry = tk.Entry(Patch_Interface, width=5, justify="center", validate="key",
                              validatecommand=(Validation, "%P", "%d"))
        WidthEntry.place(anchor="center", relx=0.5, rely=0.63)

        ExtentionLabel = tk.Label(Patch_Interface, text="Select your image output extension")
        ExtentionLabel.place(relx=0.5, rely=0.695, anchor="center")

        ExtentionVar = tk.IntVar()
        JPGExtention = tk.Radiobutton(Patch_Interface, text=".JPG", var=ExtentionVar, value=0)
        JPGExtention.place(relx=0.4, rely=0.725, anchor="center")
        PNGExtention = tk.Radiobutton(Patch_Interface, text=".PNG", var=ExtentionVar, value=1)
        PNGExtention.place(relx=0.5, rely=0.725, anchor="center")
        OrigExtention = tk.Radiobutton(Patch_Interface, text="As Orignal", var=ExtentionVar, value=2)
        OrigExtention.place(relx=0.6, rely=0.725, anchor="center")

        PreviewButton = tk.Button(Patch_Interface, text="Preview", command=PreviewGrid)
        PreviewButton.place(anchor="center", relx=0.5, rely=0.76)

        SaveLabel = tk.Label(Patch_Interface, text="Select save location", width=40)
        SaveLabel.place(anchor="center", relx=0.5, rely=0.80)

        SaveEntry = tk.Entry(Patch_Interface, width=40, bg="white", justify="center")
        SaveEntry.place(anchor="center", relx=0.475, rely=0.83)

        SaveBrowse = tk.Button(Patch_Interface, text="Browse...", command=GetSaveFolder)
        SaveBrowse.place(anchor="center", relx=0.65, rely=0.83)

        WarningLabel = tk.Label(Patch_Interface,
                                text="WARNING - Save directory is not empty. Images may not save correctly.", fg="red")

        StartButton = tk.Button(Patch_Interface, text="Start", width=8, command=PreChecks)
        StartButton.place(anchor="center", relx=0.55, rely=0.90)
        BackButton = tk.Button(Patch_Interface, text="Back", width=8)
        BackButton.place(anchor="center", relx=0.45, rely=0.90)

        Progresslabel = tk.Label(Patch_Interface, text="Progress:")
        Progresslabel.place(anchor="center", relx=0.5, rely=0.94)
        ProgressVarCol = tk.IntVar()
        pgbar = ttk.Progressbar(Patch_Interface, length=600, orient="horizontal", maximum=100, value=0,
                                variable=ProgressVarCol)
        pgbar.place(anchor="center", relx=0.5, rely=0.97)

        ProgressVarImg = tk.IntVar()
        pgbarImg = ttk.Progressbar(Patch_Interface, length=600, orient="horizontal", maximum=100, value=0,
                                   variable=ProgressVarImg)


    def White_App():
        WS_Interface = tk.Toplevel(HomeScreen)
        WS_Interface.geometry(f"1500x800+{ScreenWidthMiddle - 750}+{ScreenHeightMiddle - 400}")
        WS_Interface.wm_title("Whitespace Filter")
        WS_Interface.resizable(False, False)
        WS_Interface.protocol("WM_DELETE_WINDOW", lambda: Close_And_Home(WS_Interface))
        WS_Interface.iconbitmap(r"Icon/HistoSquare.ico")

        ChunkyFont = font.Font(size=40, slant="italic")
        Images_Processed = 0

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
                    if '.jpg' in file or '.jfif' in file or ".png" in file:
                        ImagePaths.append(os.path.join(r, file))

            ActiveImages = ImagePaths[:ShownImageLocation]
            for A in ActiveImages:
                ShownImagesPaths.append(A)

            # Make Thumbnail Images
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

            CanvasXLoc = 0

            TotalWidth = 0
            for I in ShownImagesThumbnailsSizes:
                width = I[0]
                TotalWidth += width

            DeadSpace = 880 - TotalWidth
            DeadspaceFraction = round(DeadSpace / 3)
            for I in range(len(ShownImagesThumbnails)):
                Nigel = ShownImagesThumbnails[I]
                width, height = ShownImagesThumbnailsSizes[I]
                print(width, height)
                OrigCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
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
            elif not os.path.isdir(Step1Entry.get()):
                Step1Text.config(text="Please select a valid folder", fg="red")

            if len(Step2Entry.get()) == 0:
                Step2Text.config(fg="red")

            if TypeDropdown.get() == "Positive Pixel Classification":
                if len(Binary_Option_Entry.get()) == 0:
                    Binary_Option_Label.config(fg="red")
                    Errors += 1

                elif float(Binary_Option_Entry.get()) > 1 or float(Binary_Option_Entry.get()) < 0:
                    Binary_Option_Label.config(fg="red")
                    Errors += 1
                print(len(BlurEntry.get()))
                if len(BlurEntry.get()) == 0:
                    BlurLabel.config(fg="red")
                    Errors += 1

            elif TypeDropdown.get() == "Adaptive Threshold - Mean":

                if len(BlurEntry.get()) == 0:
                    BlurLabel.config(fg="red")
                    Errors += 1

                elif int(BlurEntry.get()) % 2 == 0:
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
                        CanvasSize = 200, 200
                        outimg.thumbnail(CanvasSize, Image.ANTIALIAS)
                        PreviewImageMask = ImageTk.PhotoImage(outimg)
                        ShownImageThumbnailMasks.append(PreviewImageMask)

                    print(ShownImagePositivePercentage)
                    TotalWidth = 0
                    CanvasXLoc = 0
                    for I in ShownImagesThumbnailsSizes:
                        width = I[0]
                        TotalWidth += width

                    DeadSpace = 880 - TotalWidth
                    DeadspaceFraction = round(DeadSpace / 3)
                    for I in range(len(ShownImageThumbnailMasks)):
                        Nigel = ShownImageThumbnailMasks[I]
                        width, height = ShownImagesThumbnailsSizes[I]

                        print(width, height)
                        MaskCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
                        MaskCanvas.create_text(CanvasXLoc + 60, 250,
                                               text=f"Tissue: {str(round(ShownImagePositivePercentage[I], 2))}%",
                                               anchor="w")
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
                        CanvasSize = 200, 200
                        outimg.thumbnail(CanvasSize, Image.ANTIALIAS)
                        PreviewImageMask = ImageTk.PhotoImage(outimg)
                        ShownImageThumbnailMasks.append(PreviewImageMask)

                    print(ShownImagePositivePercentage)
                    TotalWidth = 0
                    CanvasXLoc = 0
                    for I in ShownImagesThumbnailsSizes:
                        width = I[0]
                        TotalWidth += width

                    DeadSpace = 880 - TotalWidth
                    DeadspaceFraction = round(DeadSpace / 3)
                    for I in range(len(ShownImageThumbnailMasks)):
                        Nigel = ShownImageThumbnailMasks[I]
                        width, height = ShownImagesThumbnailsSizes[I]

                        print(width, height)
                        MaskCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
                        MaskCanvas.create_text(CanvasXLoc + 60, 250,
                                               text=f"Tissue: {str(round(ShownImagePositivePercentage[I], 2))}%",
                                               anchor="w")
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
                        CanvasSize = 200, 200
                        outimg.thumbnail(CanvasSize, Image.ANTIALIAS)
                        PreviewImageMask = ImageTk.PhotoImage(outimg)
                        ShownImageThumbnailMasks.append(PreviewImageMask)

                    print(ShownImagePositivePercentage)
                    TotalWidth = 0
                    CanvasXLoc = 0
                    for I in ShownImagesThumbnailsSizes:
                        width = I[0]
                        TotalWidth += width

                    DeadSpace = 880 - TotalWidth
                    DeadspaceFraction = round(DeadSpace / 3)
                    for I in range(len(ShownImageThumbnailMasks)):
                        Nigel = ShownImageThumbnailMasks[I]
                        width, height = ShownImagesThumbnailsSizes[I]

                        print(width, height)
                        MaskCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
                        MaskCanvas.create_text(CanvasXLoc + 60, 250,
                                               text=f"Tissue: {str(round(ShownImagePositivePercentage[I], 2))}%",
                                               anchor="w")
                        CanvasXLoc += (width + DeadspaceFraction)

                elif TypeDropdown.get() == "Otsu Binarisation":

                    Blur = int(BlurEntry.get())

                    for I in ShownImagesPaths:
                        img = cv2.imread(I, 0)
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
                        CanvasSize = 200, 200
                        outimg.thumbnail(CanvasSize, Image.ANTIALIAS)
                        PreviewImageMask = ImageTk.PhotoImage(outimg)
                        ShownImageThumbnailMasks.append(PreviewImageMask)

                    print(ShownImagePositivePercentage)
                    TotalWidth = 0
                    CanvasXLoc = 0
                    for I in ShownImagesThumbnailsSizes:
                        width = I[0]
                        TotalWidth += width

                    DeadSpace = 880 - TotalWidth
                    DeadspaceFraction = round(DeadSpace / 3)
                    for I in range(len(ShownImageThumbnailMasks)):
                        Nigel = ShownImageThumbnailMasks[I]
                        width, height = ShownImagesThumbnailsSizes[I]

                        MaskCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
                        MaskCanvas.create_text(CanvasXLoc + 60, 250,
                                               text=f"Tissue: {str(round(ShownImagePositivePercentage[I], 2))}%",
                                               anchor="w")
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
                CanvasSize = 200, 200
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

            DeadSpace = 880 - TotalWidth
            DeadspaceFraction = round(DeadSpace / 3)
            for I in range(len(ShownImagesThumbnails)):
                Nigel = ShownImagesThumbnails[I]
                width, height = ShownImagesThumbnailsSizes[I]
                print(width, height)
                OrigCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
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
                CanvasSize = 200, 200
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

            DeadSpace = 880 - TotalWidth
            DeadspaceFraction = round(DeadSpace / 3)
            for I in range(len(ShownImagesThumbnails)):
                Nigel = ShownImagesThumbnails[I]
                width, height = ShownImagesThumbnailsSizes[I]
                print(width, height)
                OrigCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
                CanvasXLoc += (width + DeadspaceFraction)

            try:
                CreateMask()
            except:
                pass

        def WS_Thread_Threading():
            Errors = PreChecks()
            if Errors == 0:
                if CheckVar.get() == 1:
                    try:
                        NewFolder = os.path.join(Step1Entry.get(), "Removed Images")
                        os.mkdir(NewFolder)
                    except:
                        pass
                t = threading.Thread(target=WS_Threading)
                t.start()
                Update_Progress()

        def WS_Threading():

            pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
            for I in range(len(ImagePaths)):
                pool.submit(WS_Core, ImagePaths[I])
            pool.shutdown(wait=True)

        def WS_Core(ImageIn):
            nonlocal Images_Processed
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
                if PositivePercentage < int(Step2Entry.get()):
                    if CheckVar.get() == 1:
                        imagename = os.path.split(ImageIn)[-1]
                        DeletedLocation = os.path.join(Step1Entry.get(), "Removed Images", )
                        NewPath = os.path.join(DeletedLocation, imagename)
                        os.replace(ImageIn, NewPath)
                    elif CheckVar.get() == 0:
                        os.remove(ImageIn)

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
                    if CheckVar.get() == 1:
                        imagename = os.path.split(ImageIn)[-1]
                        DeletedLocation = os.path.join(Step1Entry.get(), "Removed Images", )
                        NewPath = os.path.join(DeletedLocation, imagename)
                        os.replace(ImageIn, NewPath)
                    elif CheckVar.get() == 0:
                        os.remove(ImageIn)

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
                    if CheckVar.get() == 1:
                        imagename = os.path.split(ImageIn)[-1]
                        DeletedLocation = os.path.join(Step1Entry.get(), "Removed Images", )
                        NewPath = os.path.join(DeletedLocation, imagename)
                        os.replace(ImageIn, NewPath)
                    elif CheckVar.get() == 0:
                        os.remove(ImageIn)

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
                    if CheckVar.get() == 1:
                        imagename = os.path.split(ImageIn)[-1]
                        DeletedLocation = os.path.join(Step1Entry.get(), "Removed Images", )
                        NewPath = os.path.join(DeletedLocation, imagename)
                        os.replace(ImageIn, NewPath)
                    elif CheckVar.get() == 0:
                        os.remove(ImageIn)

            Images_Processed += 1

        def Update_Progress():
            nonlocal Images_Processed
            nonlocal ImagePaths
            while Images_Processed < len(ImagePaths):
                print(f"{Images_Processed}/{len(ImagePaths)}")
                pgbar.config(maximum=len(ImagePaths))
                BeginButton.config(state="disabled")
                ProgressVar.set(Images_Processed)
                ProgressLabel.config(text=f"Progress: {Images_Processed} / {len(ImagePaths)} complete")
                WS_Interface.update()
                time.sleep(0.000001)

            ProgressLabel.config(text=f"Progress {Images_Processed} / {len(ImagePaths)} complete")
            ProgressVar.set(Images_Processed)
            WS_Interface.update()
            BeginButton.config(state="normal")
            Pop()

        def Pop():

            Popup = tk.Toplevel(HomeScreen)
            Popup.geometry(f"200x100+{ScreenWidthMiddle - 100}+{ScreenHeightMiddle - 50}")
            Popup.title("Complete")
            Popup.resizable(False, False)
            Popup.iconbitmap(r"Icon/HistoSquare.ico")
            Info = tk.Label(Popup, text="Process Complete\n"
                                        "Stay here or return to home screen?")

            Popup.grab_set()

            def ReturnHome():
                Popup.destroy()
                HomeScreen.deiconify()
                WS_Interface.destroy()

            def Stay():
                Popup.destroy()
                Popup.grab_release()
                ResetAll()

            Info.place(relx=0.5, rely=0.2, anchor="center")
            Home_Button = tk.Button(Popup, text="Home", command=ReturnHome, width=10)
            Home_Button.place(relx=0.25, rely=0.6, anchor="center")
            Stay_Button = tk.Button(Popup, text="Stay", command=Stay, width=10)
            Stay_Button.place(relx=0.75, rely=0.6, anchor="center")

        def ResetAll():
            nonlocal Images_Processed
            nonlocal ShownImageLocation
            Images_Processed = 0

            ImagePaths.clear()
            ShownImagesPaths.clear()
            ShownImagesThumbnails.clear()
            ShownImagesThumbnailsSizes.clear()
            ShownImageThumbnailMasks.clear()
            ShownImagePositivePercentage.clear()
            ShownImageLocation = 4

            Step1Entry.delete(0, 10000)
            Step2Entry.delete(0, 10000)
            Binary_Option_Entry.delete(0, 10000)
            BlurEntry.delete(0, 10000)
            Block_Size_Entry.delete(0, 10000)
            Anchor_Entry.delete(0, 10000)

            ProgressVar.set(0)
            ProgressLabel.config(text="")

            MaskCanvas.delete("all")
            OrigCanvas.delete("all")

        Step1Text = tk.Label(WS_Interface, fg="black",
                             text="Step 1: Select your working directory. (Root folder containing images)")
        Step1Text.place(anchor="center", relx=0.17, rely=0.05)

        Step1Entry = tk.Entry(WS_Interface, width=40)
        Step1Entry.place(anchor="center", relx=0.17, rely=0.1)

        ButtonForFiles2 = tk.Button(WS_Interface, text="Browse...", command=LoadImages)
        ButtonForFiles2.place(anchor="center", relx=0.285, rely=0.1)

        Step2Text = tk.Label(WS_Interface, fg="black", text="Step 2: Select your minimum wanted tissue coverage. (0-100)")
        Step2Text.place(anchor="center", relx=0.17, rely=0.15)

        Validation = (WS_Interface.register(IntOnly))
        EV = tk.StringVar()
        Step2Entry = tk.Entry(WS_Interface, width=5, textvariable=EV, justify="center", validate="key",
                              validatecommand=(Validation, "%P", "%d"))
        Step2Entry.place(anchor="center", relx=0.17, rely=0.2)
        Step2Entry.bind_all('<Key>', Change_Slider)

        Step2Scale = tk.Scale(WS_Interface, from_=0, to=100, orient="horizontal", showvalue=0, tickinterval=50,
                              command=Change_Tissue_Entry)
        Step2Scale.place(anchor="center", relx=0.17, rely=0.25)

        Step3Label = tk.Label(WS_Interface, text="Step 3: Select your Image Thresholding method")
        Step3Label.place(anchor="center", relx=0.17, rely=0.3)

        HelpLabel = tk.Label(WS_Interface, text="")
        HelpLabel.place(anchor="center", relx=0.17, rely=0.55)

        OptionList = ["Positive Pixel Classification", "Adaptive Threshold - Mean", "Adaptive Threshold - Gaussian",
                      "Otsu Binarisation"]
        TypeDropdown = ttk.Combobox(WS_Interface, values=OptionList, width=35, state="readonly")
        TypeDropdown.place(relx=0.17, rely=0.35, anchor="center")
        TypeDropdown.bind("<<ComboboxSelected>>", lambda e: BringUpOptions())
        TypeDropdown.set("Positive Pixel Classification")

        Binary_Option_Label = tk.Label(WS_Interface, text="Pixel Cutoff (0-1)")
        Binary_Option_Entry = tk.Entry(WS_Interface, width=4, justify="center")
        Binary_Option_Text = "The intensity of a pixel considered to be negative. A higher value includes lighter pixels."
        Binary_Option_Label.bind("<Enter>", lambda e: HelpLabel.config(text=Binary_Option_Text))
        Binary_Option_Label.bind("<Leave>", lambda e: HelpLabel.config(text=""))

        Binary_Option_Label.place(relx=0.12, rely=0.4, anchor="center")
        Binary_Option_Entry.place(relx=0.20, rely=0.4, anchor="center")

        BlurLabel = tk.Label(WS_Interface, text="Smoothing Intensity (Odd, >=0)")
        BlurEntry = tk.Entry(WS_Interface, width=4, justify="center")
        Blur_Option_Text = "Intensity of blur to add to the image. This can help fill in small gaps in the image"
        BlurLabel.bind("<Enter>", lambda e: HelpLabel.config(text=Blur_Option_Text))
        BlurLabel.bind("<Leave>", lambda e: HelpLabel.config(text=""))

        BlurLabel.place(relx=0.12, rely=0.45, anchor="center")
        BlurEntry.place(relx=0.20, rely=0.45, anchor="center")

        Info_Label = tk.Label(WS_Interface, text="Pixel classed as positive or negative based on a single threshold.\n"
                                                 "The higher the threshold, the lighter the pixels included.")
        Info_Label.place(anchor="center", relx=0.17, rely=0.6)

        Block_Size_Label = tk.Label(WS_Interface, text="Block Size (Odd, >0)")
        Block_Size_Entry = tk.Entry(WS_Interface, width=4, justify="center")
        Block_Info = "Kernel size for the image. Must be an odd positive integer"
        Block_Size_Label.bind("<Enter>", lambda e: HelpLabel.config(text=Block_Info))
        Block_Size_Label.bind("<Leave>", lambda e: HelpLabel.config(text=""))

        Anchor_Label = tk.Label(WS_Interface, text="Anchor Value (>1)")
        Anchor_Entry = tk.Entry(WS_Interface, width=4, justify="center")
        Anchor_Info = "The value subtracted from the mean or weighed mean calculated. Must be a positive integer"
        Anchor_Label.bind("<Enter>", lambda e: HelpLabel.config(text=Anchor_Info))
        Anchor_Label.bind("<Leave>", lambda e: HelpLabel.config(text=""))

        Preview_Button = tk.Button(WS_Interface, text="Preview", command=CreateMask)
        Preview_Button.place(anchor="center", relx=0.17, rely=0.5)

        CheckVar = tk.IntVar(value=1)
        KeepChecker = tk.Checkbutton(WS_Interface, variable=CheckVar, offvalue=0, onvalue=1,
                                     text="Keep sub-threshold images in separate folder?")
        KeepChecker.place(anchor="center", relx=0.17, rely=0.725)

        OrigCanvas = tk.Canvas(WS_Interface, height=220, width=880)
        OrigCanvas.place(anchor="n", relx=0.65, rely=0.05)

        MaskCanvas = tk.Canvas(WS_Interface, height=300, width=880)
        MaskCanvas.place(anchor="n", relx=0.65, rely=0.4)

        ProgressVar = tk.IntVar()
        pgbar = ttk.Progressbar(WS_Interface, length=500, orient="horizontal", maximum=100, value=0, variable=ProgressVar)
        pgbar.place(anchor="center", relx=0.65, rely=0.95)

        Prev_Img_Button = tk.Button(WS_Interface, text="Previous Set", command=PrevPreview, width=11)
        Prev_Img_Button.place(anchor="center", relx=0.6, rely=0.8)

        Next_Img_Button = tk.Button(WS_Interface, text="Next Set", command=NextPreview, width=11)
        Next_Img_Button.place(anchor="center", relx=0.7, rely=0.8)

        BeginButton = tk.Button(WS_Interface, text="Begin Removal", height=1, width=15, command=WS_Thread_Threading)
        BeginButton.place(anchor="center", relx=0.65, rely=0.9)

        pgcount = tk.Label(WS_Interface, text="")
        pgcount.place(anchor="center", relx=0.2, rely=0.94)

        ProgressLabel = tk.Label(WS_Interface, text="")
        ProgressLabel.place(anchor="center", relx=0.65, rely=0.98)


    def Balance_App():
        Balance_Interface = tk.Toplevel(HomeScreen)
        Balance_Interface.geometry(f"500x600+{ScreenWidthMiddle - 250}+{ScreenHeightMiddle - 300}")
        Balance_Interface.wm_title("Dataset Balancing")
        Balance_Interface.resizable(False, False)
        Balance_Interface.iconbitmap(r"Icon/HistoSquare.ico")
        Balance_Interface.protocol("WM_DELETE_WINDOW", lambda: Close_And_Home(Balance_Interface))

        Images_Processed = 0
        Images_To_Process = 0

        SetOptions = 2
        BottomEdge = 210
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

        def NameSet(ButtonNumber):
            ImageList = []

            OptionsNumber = int(ButtonNumber[-1]) - 1
            try:
                ImageListOfLists.pop(OptionsNumber)
                FolderLocations.pop(OptionsNumber)
            except:
                print("Not Present")

            SelectedEntry = SetEntries[OptionsNumber]
            print(SelectedEntry)
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
            print(ImageListOfLists)
            print(len(ImageListOfLists))
            print(FolderLocations)
            print(len(FolderLocations))

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

        def AddOption():
            nonlocal SetOptions
            nonlocal BottomEdge
            TopEdge = BottomEdge + 20
            print(BottomEdge)

            SetOptions += 1

            SetFrameInner = tk.Frame(SetCanvas, height=100, width=350, relief="groove", borderwidth=1)

            SetLabel = tk.Label(SetFrameInner, text=f"Image Set {SetOptions}", name=f"label{SetOptions}")
            SetLabel.place(relx=0.5, rely=0.2, anchor="center")
            LabelList.append(SetLabel)

            FolderEntry = tk.Entry(SetFrameInner, width=40)
            FolderEntry.place(anchor="center", relx=0.4, rely=0.5)
            SetEntries.append(FolderEntry)

            ButtonForFiles = tk.Button(SetFrameInner, text="Browse...", name=f"button{SetOptions}",
                                       command=lambda: NameSet(str(ButtonForFiles)))
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

        def Update_pgbarB():
            nonlocal Images_Processed
            nonlocal Images_To_Process

            while Images_Processed < Images_To_Process:
                BeginButtonB.config(state="disabled")
                ProgressVarB.set(Images_Processed)
                ProgressLabel.config(text=f"Images Processed : {Images_Processed}/{Images_To_Process}")
                Balance_Interface.update()
                time.sleep(0.000001)

            ProgressLabel.config(text=f"Images Processed: {Images_Processed} / {Images_To_Process}")
            ProgressVarB.set(Images_Processed)
            Balance_Interface.update()
            BeginButtonB.config(state="normal")
            Pop()

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
                pgbar.config(maximum=ImagesToProcess)

                MakeBalanceLists()

                print(f"Images to Process {ImagesToProcess}")

        def MakeBalanceLists():
            nonlocal SizeList

            print(DifferenceList)
            for Y in range(len(SizeList)):
                UniqueID = 0
                ImageSet = ImageListOfLists[Y]
                ValueToChange = DifferenceList[Y]

                if ValueToChange >= 0:
                    if ValueToChange > len(ImageSet):
                        Images_For_Balancing = random.choices(ImageSet, k=ValueToChange)
                        for i in Images_For_Balancing:
                            ImagesToAdd.append(i)
                            BalanceImageNames.append(f"Balancing Image{UniqueID}")
                            UniqueID += 1
                    elif ValueToChange < len(ImageSet):
                        Images_For_Balancing = random.sample(ImageSet, k=ValueToChange)
                        for i in Images_For_Balancing:
                            ImagesToAdd.append(i)
                            BalanceImageNames.append(f"Balancing Image{UniqueID}")
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

            print(f"SizeList: {SizeList}")
            print(f"DifferenceList: {DifferenceList}")
            print(f"Images to Remove: {len(ImagesToRemove)} {ImagesToRemove}")
            print(f"Images to Add:{len(ImagesToAdd)} {ImagesToAdd}")

            AddBalanceFolders()
            Balance_Thread_Threading()
            Update_pgbarB()

        def RemoveImages(Image):
            nonlocal Images_Processed
            os.remove(Image)
            Images_Processed += 1
            print(Images_Processed)

        def AddImages(Image, OutName):
            nonlocal Images_Processed

            Rotation = [90, 180, 270]
            Flip = [1, 2]
            Angle = random.choice(Rotation)
            FlipType = random.choice(Flip)
            Image_in = imageio.imread(Image)

            ImageDir = os.path.split(Image)[0]
            ImageName = os.path.split(Image)[-1]

            if FlipType == 1:
                flip = iaa.Fliplr(1)
            else:
                flip = iaa.Flipud(1)

            Augments = iaa.Sequential([iaa.Affine(rotate=Angle), flip])

            Image_out = Augments.augment_image(Image_in)

            for f in range(len(FolderLocations)):
                if FolderLocations[f] in ImageDir:
                    DestFolder = BalanceFolderLocations[f]
                    imageio.imwrite(os.path.join(str(DestFolder), f"{OutName} {ImageName}"), Image_out)
            Images_Processed += 1

        def AddBalanceFolders():
            for f in range(len(FolderLocations)):
                if DifferenceList[f] <= 0:
                    pass
                else:
                    BalancedFolder = os.path.join(FolderLocations[f], "Balanced Images")
                    BalanceFolderLocations.append(BalancedFolder)
                    try:
                        os.mkdir(BalancedFolder)
                    except FileExistsError:
                        print("Folder already exists")

        def Balance_Thread_Threading():
            t = threading.Thread(target=Balance_Threading)
            t.start()

        def Balance_Threading():
            if len(ImagesToRemove) > 0:
                pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
                for I in range(len(ImagesToRemove)):
                    pool.submit(RemoveImages, ImagesToRemove[I])
                pool.shutdown(wait=True)

            if len(ImagesToAdd) > 0:
                pool = ThreadPoolExecutor(max_workers=ThreadsAllowed)
                for I in range(len(ImagesToAdd)):
                    pool.submit(AddImages, ImagesToAdd[I], BalanceImageNames[I])
                pool.shutdown(wait=True)

        def Scrolly(e):
            SetCanvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def Pop():

            Popup = tk.Toplevel(HomeScreen)
            Popup.geometry(f"200x100+{ScreenWidthMiddle - 100}+{ScreenHeightMiddle - 50}")
            Popup.title("Complete")
            Popup.resizable(False, False)
            Popup.iconbitmap(r"Icon/HistoSquare.ico")
            Info = tk.Label(Popup, text="Process Complete\n"
                                        "Stay here or return to home screen?")

            Popup.grab_set()

            def ReturnHome():
                Popup.destroy()
                HomeScreen.deiconify()
                Balance_Interface.destroy()

            def Stay():
                Popup.destroy()
                Popup.grab_release()
                ResetAll()

            Info.place(relx=0.5, rely=0.2, anchor="center")
            Home_Button = tk.Button(Popup, text="Home", command=ReturnHome, width=10)
            Home_Button.place(relx=0.25, rely=0.6, anchor="center")
            Stay_Button = tk.Button(Popup, text="Stay", command=Stay, width=10)
            Stay_Button.place(relx=0.75, rely=0.6, anchor="center")

        def ResetAll():

            nonlocal Images_Processed
            nonlocal Images_To_Process
            nonlocal SetOptions
            nonlocal BottomEdge
            nonlocal SetList
            nonlocal LabelList
            nonlocal BrowseButtons
            nonlocal SetEntries
            nonlocal CountList
            nonlocal ImageListOfLists
            nonlocal FolderLocations
            nonlocal BalanceFolderLocations
            nonlocal BalanceImageNames
            nonlocal SizeList
            nonlocal DifferenceList
            nonlocal ImagesToAdd
            nonlocal ImagesToRemove

            for E in SetEntries:
                E.delete(0, 10000)

            for C in CountList:
                C.config(text="")

            Update_Label.config(text="")
            MinMaxLabel.config(text="")
            ProgressVarB.set(0)

            Images_Processed = 0
            Images_To_Process = 0
            SetOptions = 2
            BottomEdge = 210

            while len(SetList) > 2:
                SetCanvas.delete(SetList[-1])
                del SetList[-1]
                del LabelList[-1]
                del BrowseButtons[-1]
                del SetEntries[-1]

            CountList.clear()
            ImageListOfLists.clear()
            FolderLocations.clear()
            BalanceFolderLocations.clear()
            BalanceImageNames.clear()
            SizeList.clear()
            DifferenceList.clear()
            ImagesToAdd.clear()
            ImagesToRemove.clear()

            print(CountList)
            print(ImageListOfLists)
            print(FolderLocations)
            print(BalanceFolderLocations)
            print(BalanceImageNames)
            print(SizeList)
            print(DifferenceList)
            print(ImagesToAdd)
            print(ImagesToRemove)

            print(SetList)
            print(LabelList)
            print(BrowseButtons)
            print(SetEntries)

            SetCanvas.config(scrollregion=(0, 0, 400, 300))
            ProgressLabel.config(text="")

        ######################################### Place Balance Image Widgets #######################################

        UppText = tk.Label(Balance_Interface, fg="black", text=f"Here you can quick-balance your data set.")
        UppText.place(anchor="center", relx=0.5, rely=0.05)

        SetFrameOutter = tk.Frame(Balance_Interface, width=400, height=250, highlightthickness=0)
        SetFrameOutter.place(anchor="center", relx=0.5, rely=0.35)
        SetCanvas = tk.Canvas(SetFrameOutter, width=400, height=300, bg="gray86", scrollregion=(0, 0, 400, 300),
                              highlightthickness=0)
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

        Add_Button = tk.Button(Balance_Interface, text="Add Set", width=10, command=AddOption)
        Add_Button.place(anchor="center", relx=0.4, rely=0.63)
        Remove_Button = tk.Button(Balance_Interface, text="Remove Set", command=RemoveOption)
        Remove_Button.place(anchor="center", relx=0.6, rely=0.63)

        VarBal = tk.IntVar()
        Bal_Up = tk.Radiobutton(Balance_Interface, text="Balance Up", variable=VarBal, value=0, command=BalanceInfo)
        Bal_Up.place(anchor="center", relx=0.25, rely=0.78)
        Bal_Down = tk.Radiobutton(Balance_Interface, text="Balance Down", variable=VarBal, value=1, command=BalanceInfo)
        Bal_Down.place(anchor="center", relx=0.5, rely=0.78)
        Bal_Avg = tk.Radiobutton(Balance_Interface, text="Balance to Average", variable=VarBal, value=2,
                                 command=BalanceInfo)
        Bal_Avg.place(anchor="center", relx=0.75, rely=0.78)

        MinMaxLabel = tk.Label(Balance_Interface, text="", fg="blue")
        MinMaxLabel.place(anchor="center", relx=0.5, rely=0.71)

        BeginButtonB = tk.Button(Balance_Interface, text="Begin Balancing", height=1, width=15, command=Adjust_ProgressBar)
        BeginButtonB.place(anchor="center", relx=0.5, rely=0.88)

        Update_Label = tk.Label(Balance_Interface, fg="blue", text="")
        Update_Label.place(anchor="center", relx=0.5, rely=0.82)

        ProgressLabel = tk.Label(Balance_Interface, fg="black", text="")
        ProgressLabel.place(anchor="center", relx=0.5, rely=0.93)

        ProgressVarB = tk.IntVar()
        pgbar = ttk.Progressbar(Balance_Interface, length=300, orient="horizontal", maximum=100, value=0,
                                variable=ProgressVarB)
        pgbar.place(anchor="center", relx=0.5, rely=0.97)


    def Normalisation_App():
        Norm_Interface = tk.Toplevel(HomeScreen)
        Norm_Interface.title(f" Image Normalisation")
        Norm_Interface.geometry(f"1600x600+{ScreenWidthMiddle - 800}+{ScreenHeightMiddle - 300}")
        Norm_Interface.iconbitmap(r"Icon/HistoSquare.ico")
        Norm_Interface.resizable(False, False)
        Norm_Interface.protocol("WM_DELETE_WINDOW", lambda: Close_And_Home(Norm_Interface))

        TargetImage = []
        ImagePaths = []
        ShownImagesPaths = []
        ShownImagesThumbnails = []
        NormImageThubnails = []
        ShownImagesThumbnailsSizes = []
        ShownImageLocation = 4
        Processed_Images = 0

        ########################################### Place Normalisation Widgets ###########################################
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
            CanvasSize = 200, 200
            Resized_Target.thumbnail(CanvasSize, Image.ANTIALIAS)
            img = ImageTk.PhotoImage(Resized_Target)
            TargetImage.append(img)

            for I in TargetImage:
                Norm_Image_Canvas.create_image(100, 100, image=I, anchor="center")

            if len(ShownImagesThumbnails) != 0:
                LoadPreview()

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
                CanvasSize = 200, 200
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
                OrigCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
                CanvasXLoc += (width + DeadspaceFraction)

            if len(TargetImage) != 0:
                LoadPreview()

        def LoadPreview():
            NormCanvas.delete("all")
            NormImageThubnails.clear()

            for I in ShownImagesPaths:
                if TypeDropdown.get() == "Histogram Equalisation":
                    Canvas_Size = 200, 200
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
                    Canvas_Size = 200, 200
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
                NormCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
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
                CanvasSize = 200, 200
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
                OrigCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
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
                CanvasSize = 200, 200
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
                OrigCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
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
            nonlocal Processed_Images
            # print(ImagePath)
            Orig_Image = io.imread(ImagePath)

            Norm_Image = match_histograms(Orig_Image, Target, multichannel=True)

            if NormKeep_Var.get() == 1:
                Chop = ImagePath.replace(Step2Entry.get(), "")
                NewLoc = str(NewSetEntry.get()) + Chop
                print(NewLoc)
                io.imsave(NewLoc, Norm_Image)
                Processed_Images += 1

            else:
                io.imsave(ImagePath, Norm_Image)
                Processed_Images += 1

            print(Processed_Images)

        def ReinHist(ImagePath):
            nonlocal Processed_Images

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

            image_avg, image_std = getavgstd(image)
            # print(image_avg)
            original_avg, original_std = getavgstd(original)
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
                Processed_Images += 1
                print(Processed_Images)
                Chop = ImagePath.replace(Step2Entry.get(), "")
                NewLoc = str(NewSetEntry.get()) + Chop
                # print(NewLoc)
                # print(NewSetEntry.get())
                norm_image.save(NewLoc)


            else:
                Processed_Images += 1
                print(Processed_Images)
                print(ImagePath)
                norm_image.save(ImagePath)
                # Processed_Images += 1

        def Norm_Thread_Threading():
            nonlocal Processed_Images
            Errors = PreChecks()

            if Errors == 0:
                if NormKeep_Var.get() == 1:
                    RecreateFolderStructure()
                t = threading.Thread(target=Norm_Threading)
                t.start()
                Update_pgbarB()

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

        def Update_pgbarB():
            nonlocal Processed_Images
            pgbar.config(maximum=len(ImagePaths))
            while Processed_Images < len(ImagePaths):
                BeginButton.config(state="disabled")
                ProgressVarN.set(Processed_Images)
                pgbarlabel.config(text=f"Images Processed : {Processed_Images}/{len(ImagePaths)}")
                Norm_Interface.update()
                time.sleep(0.000001)

            pgbarlabel.config(text=f"Images Processed: {Processed_Images} / {len(ImagePaths)}")
            ProgressVarN.set(Processed_Images)
            Norm_Interface.update()
            BeginButton.config(state="normal")
            Pop()

        def Pop():

            Popup = tk.Toplevel(HomeScreen)
            Popup.geometry(f"200x100+{ScreenWidthMiddle - 100}+{ScreenHeightMiddle - 50}")
            Popup.title("Complete")
            Popup.resizable(False, False)
            Popup.iconbitmap(r"Icon/HistoSquare.ico")
            Info = tk.Label(Popup, text="Process Complete\n"
                                        "Stay here or return to home screen?")

            Popup.grab_set()

            def ReturnHome():
                Popup.destroy()
                HomeScreen.deiconify()
                Norm_Interface.destroy()

            def Stay():
                Popup.destroy()
                Popup.grab_release()
                ResetAll()

            Info.place(relx=0.5, rely=0.2, anchor="center")
            Home_Button = tk.Button(Popup, text="Home", command=ReturnHome, width=10)
            Home_Button.place(relx=0.25, rely=0.6, anchor="center")
            Stay_Button = tk.Button(Popup, text="Stay", command=Stay, width=10)
            Stay_Button.place(relx=0.75, rely=0.6, anchor="center")

        def ResetAll():
            nonlocal ShownImageLocation
            nonlocal Processed_Images
            TargetImage.clear()
            ImagePaths.clear()
            ShownImagesPaths.clear()
            ShownImagesThumbnails.clear()
            NormImageThubnails.clear()
            ShownImagesThumbnailsSizes.clear()
            ShownImageLocation = 4
            Processed_Images = 0
            pgbarlabel.config(text="")
            Step1Entry.delete(0, 10000)
            Step2Entry.delete(0, 10000)
            NewSetEntry.delete(0, 10000)
            ProgressVarN.set(0)

        def RecreateFolderStructure():
            OriginalTopFolder = Step2Entry.get()
            NewSetTopFolder = NewSetEntry.get()

            for In in ImagePaths:
                OriginalImageName = (os.path.split(In)[1]).split(".")[0]
                OriginalImageLocation = os.path.split(In)[0]
                OriginalImageShortLocation = OriginalImageLocation.replace(OriginalTopFolder, "")
                # os.path.join kept fucking up here, no idea why.
                NewFolder = str(NewSetTopFolder) + str(OriginalImageShortLocation)  # + "\\" + str(OriginalImageName)

                try:
                    os.makedirs(NewFolder)
                except Exception as e:
                    pass

        Step1Text = tk.Label(Norm_Interface, fg="black", text="Step 1: Please select the image you want to normalise to:")
        Step1Text.place(anchor="center", relx=0.15, rely=0.05)

        Step1Entry = tk.Entry(Norm_Interface, width=40)
        Step1Entry.place(anchor="center", relx=0.13, rely=0.1)

        ButtonForFiles = tk.Button(Norm_Interface, text="Browse...", command=GetImage)
        ButtonForFiles.place(anchor="center", relx=0.23, rely=0.1)

        Step2Text = tk.Label(Norm_Interface, fg="black", text="Step 2: Please select images for normalisation:")
        Step2Text.place(anchor="center", relx=0.15, rely=0.17)

        Step2Entry = tk.Entry(Norm_Interface, width=40)
        Step2Entry.place(anchor="center", relx=0.13, rely=0.22)

        ButtonForFiles2 = tk.Button(Norm_Interface, text="Browse...", command=LoadImages)
        ButtonForFiles2.place(anchor="center", relx=0.23, rely=0.22)

        Norm_Image_Label = tk.Label(Norm_Interface, text="Target Image:")
        Norm_Image_Label.place(relx=0.15, rely=0.27, anchor="center")

        Norm_Image_Canvas = tk.Canvas(Norm_Interface, width=200, height=200)
        Norm_Image_Canvas.place(anchor="center", relx=0.15, rely=0.5)

        OrigCanvas = tk.Canvas(Norm_Interface, width=1000, height=200)
        OrigCanvas.place(anchor="center", relx=0.65, rely=0.2)

        NormCanvas = tk.Canvas(Norm_Interface, width=1000, height=200)
        NormCanvas.place(anchor="center", relx=0.65, rely=0.6)

        NormKeep_Var = tk.IntVar(value=1)
        KeepImageCheck = tk.Checkbutton(Norm_Interface, variable=NormKeep_Var, offvalue=0, onvalue=1,
                                        text="Keep Original Images?",
                                        command=NewSetOptions)
        KeepImageCheck.place(anchor="center", relx=0.15, rely=0.8)

        NewSetLabel = tk.Label(Norm_Interface, text="Select save folder for new images")
        NewSetEntry = tk.Entry(Norm_Interface, width=40)
        NewSetButton = tk.Button(Norm_Interface, text="Browse...", command=GetSaveLocation)

        NewSetLabel.place(anchor="center", relx=0.15, rely=0.85)
        NewSetEntry.place(anchor="center", relx=0.13, rely=0.9)
        NewSetButton.place(anchor="center", relx=0.23, rely=0.9)

        TypeLabel = tk.Label(Norm_Interface, text="Select Normalisation Type")
        TypeLabel.place(anchor="center", relx=0.15, rely=0.7)

        OptionList = ["Histogram Equalisation", "Reinhard Method"]
        TypeDropdown = ttk.Combobox(Norm_Interface, values=OptionList, width=35, state="readonly")
        TypeDropdown.set(OptionList[0])
        TypeDropdown.place(anchor="center", relx=0.15, rely=0.75)
        TypeDropdown.bind("<<ComboboxSelected>>", lambda e: LoadPreview())

        WarnLabel = tk.Label(Norm_Interface, text="", fg="red")
        WarnLabel.place(anchor="center", relx=0.15, rely=0.92)

        pgbarlabel = tk.Label(Norm_Interface, text="")
        pgbarlabel.place(anchor="center", relx=0.65, rely=0.9)

        ProgressVarN = tk.IntVar()
        pgbar = ttk.Progressbar(Norm_Interface, length=500, orient="horizontal", maximum=100, value=0,
                                variable=ProgressVarN)
        pgbar.place(anchor="center", relx=0.65, rely=0.95)

        # pgcount = tk.Label(Norm_Interface, text="")
        # pgcount.place(anchor="center", relx=0.17, rely=0.95)

        Prev_Img_Button = tk.Button(Norm_Interface, text="Previous Set", width=11, command=PrevSet)
        Prev_Img_Button.place(anchor="center", relx=0.6, rely=0.85)

        Next_Img_Button = tk.Button(Norm_Interface, text="Next Set", width=11, command=NextSet)
        Next_Img_Button.place(anchor="center", relx=0.675, rely=0.85)

        Begin_Label = tk.Label(Norm_Interface, fg="black", text="Step 3: Press to begin -->")
        Begin_Label.place(anchor="center", relx=0.78, rely=0.85)

        BeginButton = tk.Button(Norm_Interface, text="Begin Normalisation", height=1, width=15,
                                command=Norm_Thread_Threading)
        BeginButton.place(anchor="center", relx=0.87, rely=0.85)


    def Augmentation_App():
        # Variables For adding options
        ShownImageLocation = 4
        ImageCount = 0
        Processed_Images = 0

        ActiveOption = 0

        AugOptions = -1
        ActiveAugmentations = -1
        Yplace = 10
        BottomEdge = 0
        InnerFramesList = []
        OptionWidgetListID = []
        LabelList = []
        OPButtList = []
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

        def AddOption():
            nonlocal AugOptions
            nonlocal Yplace
            nonlocal BottomEdge
            nonlocal ActiveAugmentations

            ActiveVar = tk.IntVar()
            # print(f"Type = {(type(ActiveVar))}")
            AugOptions += 1
            ActiveAugmentations += 1
            OptionHistory.append("")
            SubOptionHistory.append("")
            SavedOptionsList.append("")
            SavedOptionCommand.append([])
            SavedWidgetValueList.append([])

            AugFrameInner = tk.Frame(AugCanvas, height=100, width=480, relief="raised", borderwidth=1, bg="gray88")
            InnerFramesList.append(AugFrameInner)

            AugLabel = tk.Label(AugFrameInner, text=f"Augmentation {AugOptions + 1}", name=f"label{AugOptions}",
                                bg="gray88")
            AugLabel.place(relx=0.5, rely=0.2, anchor="center")
            LabelList.append(AugLabel)

            DescriptionLabel = tk.Label(AugFrameInner, text="", name=f"desclabel{AugOptions}", bg="gray88")
            DescriptionLabel.place(relx=0.5, rely=0.8, anchor="center")
            DescriptionList.append(DescriptionLabel)

            OptionButton = tk.Button(AugFrameInner, text="Options", name=f"options{AugOptions}",
                                     command=lambda: OpenOption(str(OptionButton)), bg="gray88")
            OptionButton.place(relx=0.85, rely=0.5, anchor="center")
            OPButtList.append(OptionButton)

            OptionList = ["Colour Options", "Brightness and Contrast", "Sharpen and Emboss", "Noise", "Dropout", "Blur",
                          "Geometric Transformations", "Edge Detection", "Colour Segmentation", "Artistic Options"]
            TypeDropdown = ttk.Combobox(AugFrameInner, values=OptionList, width=35, state="readonly",
                                        name=f"dropdown{AugOptions}")

            TypeDropdown.set("Select Augmentation Type")
            TypeDropdown.place(relx=0.5, rely=0.5, anchor="center")
            TypeDropDownList.append(TypeDropdown)
            TypeDropdown.bind("<<ComboboxSelected>>", lambda e: OpenOption(str(OptionButton)))

            ID = AugCanvas.create_window(250, Yplace, window=AugFrameInner, anchor="n")
            Yplace += 120
            BottomEdge += 120
            if BottomEdge > 300:
                AugCanvas.config(scrollregion=(0, 0, 500, BottomEdge))
            OptionWidgetListID.append(ID)

        def RemoveOption():
            nonlocal AugOptions
            nonlocal Yplace
            nonlocal BottomEdge
            nonlocal ActiveAugmentations
            nonlocal ActiveOption
            AugCanvas.delete(OptionWidgetListID[-1])

            if len(OptionHistory) - 1 == ActiveOption:
                for x in ActiveWidgetList:
                    x.place_forget()
                InfoLabel.config(text="")

            del OptionWidgetListID[-1]
            del TypeDropDownList[-1]
            del LabelList[-1]
            del OPButtList[-1]
            del InnerFramesList[-1]
            del DescriptionList[-1]
            del SubOptionHistory[-1]
            del SavedOptionsList[-1]
            del SavedOptionCommand[-1]
            del SavedWidgetValueList[-1]
            del OptionHistory[-1]

            AugOptions -= 1
            Yplace -= 120
            BottomEdge -= 120
            print(AugOptions)
            if BottomEdge > 300:
                AugCanvas.config(scrollregion=(0, 0, 500, BottomEdge))
            else:
                AugCanvas.config(scrollregion=(0, 0, 500, 300))
            ActiveAugmentations -= 1

            for x in SavedOptionCommand:
                print(x)

            print(OptionHistory)
            print(SubOptionHistory)
            print(SavedWidgetValueList)

        def OpenOption(ButtonNumber):
            nonlocal ActiveOption

            SubDescriptionLabel.config(text="")

            for x in ActiveWidgetList:
                x.place_forget()

            DescLabel.config(text="")
            Num = int(ButtonNumber[-1])  # Identify which options button was pressed
            print(Num)

            ActiveOption = Num

            for I in range(len(InnerFramesList)):
                InnerFramesList[I].config(relief="raised", bg="gray88")
                LabelList[I].config(bg="gray88")
                DescriptionList[I].config(bg="gray88")

            InnerFramesList[Num].config(relief="sunken", bg="gray64")
            LabelList[Num].config(bg="gray64")
            DescriptionList[Num].config(bg="gray64")
            LabelList[Num].config(fg="black", text=f"Augmnetation {Num + 1}")

            try:
                del OptionHistory[Num]
            except:
                pass
            OptionHistory.insert(Num, TypeDropDownList[Num].get())

            Augtype = TypeDropDownList[Num].get()

            try:
                LastSub = SubOptionHistory[Num]
            except:
                print("No History for this detected")
                LastSub = ""

            if Augtype == "Select Augmentation Type":
                LabelList[Num].config(fg="red", text="Please select annotation option")
            else:
                # print(f"Showing Options for {Augtype}")
                pass

            if Augtype == "Colour Options":
                DisplayColourOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Brightness and Contrast":
                DisplayBriConOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Sharpen and Emboss":
                DisplaySharpEmbOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Noise":
                DisplayNoiseOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Dropout":
                DisplayDropoutOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Blur":
                DisplayBlurOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Geometric Transformations":
                print("Correct")
                DisplayGeoOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Edge Detection":
                DisplayEdgeOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Colour Segmentation":
                DisplayColourSegOptions(Num, OptionsFrame, LastSub)

            if Augtype == "Artistic Options":
                DisplayArtSegOptions(Num, OptionsFrame, LastSub)

            ##################################### Option Displays ############################################

        def PreviewAugmentations():
            Aug2Label = tk.Label(Aug_Interface, text="Augmented Images", font=20)
            Aug2Label.place(anchor="center", relx=0.7, rely=0.37)
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
                CanvasSize = 220, 220
                Image_Out.thumbnail(CanvasSize, Image.ANTIALIAS)
                PreviewImage = ImageTk.PhotoImage(Image_Out)
                PreviewImageThumbnails.append(PreviewImage)

            CanvasXLoc = 0

            TotalWidth = 0
            for I in ShownImagesThumbnailsSizes:
                width = I[0]
                TotalWidth += width

            DeadSpace = 1000 - TotalWidth
            DeadspaceFraction = round(DeadSpace / 3)
            for I in range(len(PreviewImageThumbnails)):
                Nigel = PreviewImageThumbnails[I]
                width, height = ShownImagesThumbnailsSizes[I]
                PrevCanvas.create_image(CanvasXLoc, 125, image=Nigel, anchor="w")
                CanvasXLoc += (width + DeadspaceFraction)

        def DisplayColourOptions(Number, Frame, LastSub):
            No = Number

            ColourOptions = ["Invert Colours", "RGB Levels", "Saturation", "Greyscale",
                             "Temprature", "K-means Colour Quantization"]
            SubTypeDropdown = ttk.Combobox(Frame, values=ColourOptions, width=35, state="readonly",
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
                    print(SavedWidgetValueList)

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
                        AugList.append(iaa.WithChannels(2, iaa.Add((int(GreenEntryMin.get()), int(GreenEntryMax.get())),
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
                    RedLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    RedLabelMin.config(text="Minimum:")
                    RedLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    RedLabelMax.config(text="Maximum:")
                    RedEntryMin.place(relx=0.45, rely=0.27, anchor="center")  # Red min entry
                    RedEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry

                    GreenLevelCheck.place(relx=0.5, rely=0.4, anchor="center")
                    GreenLevelCheck.config(text="Green", command=ActivateGreen)
                    GreenLevelVar.set(1)
                    GreenLabelMin.place(relx=0.35, rely=0.47, anchor="center")
                    GreenLabelMin.config(text="Minimum:")
                    GreenLabelMax.place(relx=0.57, rely=0.47, anchor="center")
                    GreenLabelMax.config(text="Maximum:")
                    GreenEntryMin.place(relx=0.45, rely=0.47, anchor="center")  # Green max entry
                    GreenEntryMax.place(relx=0.67, rely=0.47, anchor="center")  # Green max entry

                    BlueLevelCheck.place(relx=0.5, rely=0.6, anchor="center")
                    BlueLevelCheck.config(text="Blue", command=ActivateBlue)
                    BlueLevelVar.set(1)
                    BlueLabelMin.place(relx=0.35, rely=0.67, anchor="center")
                    BlueLabelMin.config(text="Minimum:")
                    BlueLabelMax.place(relx=0.57, rely=0.67, anchor="center")
                    BlueLabelMax.config(text="Maximum:")
                    BlueEntryMin.place(relx=0.45, rely=0.67, anchor="center")  # Blue max entry
                    BlueEntryMax.place(relx=0.67, rely=0.67, anchor="center")  # Blue max entry

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

                    SatLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    SatLabelMin.config(text="Minimum:")
                    SatLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    SatLabelMax.config(text="Maximum:")
                    SatEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    SatEntryMin.config(state="normal")
                    SatEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
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

                    GrayLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    GrayLabelMin.config(text="Minimum:")
                    GrayLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    GrayLabelMax.config(text="Maximum:")
                    GrayEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    GrayEntryMax.place(relx=0.67, rely=0.27, anchor="center")

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

                    TempLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    TempLabelMin.config(text="Minimum:")
                    TempLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    TempLabelMax.config(text="Maximum:")
                    TempEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    TempEntryMax.place(relx=0.67, rely=0.27, anchor="center")

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

                    KMeanLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    KMeanLabelMin.config(text="Minimum:")
                    KMeanLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    KMeanLabelMax.config(text="Maximum:")
                    KMeanEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    KMeanEntryMax.place(relx=0.67, rely=0.27, anchor="center")

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
                print(SavedWidgetValueList)

            if SubOptionHistory[Number] != "Select Colour adjustment option":
                try:
                    SubTypeDropdown.set(SubOptionHistory[Number])
                except:
                    SubTypeDropdown.set("Select Colour adjustment option")

            DisplaySubOptions()

            ###################################################################################################

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

            SigConGainLable = tk.Label(Frame, text="Enter the gain. Use whole values between 1 and 10")
            SigConEntryMinGain = tk.Entry(Frame, width=6)
            SigConEntryMaxGain = tk.Entry(Frame, width=6)
            SigConLabelMinGain = tk.Label(Frame, text="", bg="gray86")
            SigConLabelMaxGain = tk.Label(Frame, text="", bg="gray86")

            SigConCutoffLabel = tk.Label(Frame, text="Adjust cutoff value between 0 and 1")
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

                    BriLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    BriLabelMin.config(text="Minimum:")
                    BriLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    BriLabelMax.config(text="Maximum:")
                    BriEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    BriEntryMin.config(state="normal")
                    BriEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
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

                    GamConLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    GamConLabelMin.config(text="Minimum:")
                    GamConLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    GamConLabelMax.config(text="Maximum:")
                    GamConEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    GamConEntryMin.config(state="normal")
                    GamConEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
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
                    SigConLabelMinCutoff.place(relx=0.35, rely=0.27, anchor="center")
                    SigConLabelMinCutoff.config(text="Minimum:")
                    SigConLabelMaxCutoff.place(relx=0.57, rely=0.27, anchor="center")
                    SigConLabelMaxCutoff.config(text="Maximum:")
                    SigConEntryMinCutoff.place(relx=0.45, rely=0.27, anchor="center")
                    SigConEntryMinCutoff.config(state="normal")
                    SigConEntryMaxCutoff.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
                    SigConEntryMaxCutoff.config(state="normal")

                    SigConGainLable.place(anchor="center", relx=0.5, rely=0.38)
                    SigConLabelMinGain.place(relx=0.35, rely=0.45, anchor="center")
                    SigConLabelMinGain.config(text="Minimum:")
                    SigConLabelMaxGain.place(relx=0.57, rely=0.45, anchor="center")
                    SigConLabelMaxGain.config(text="Maximum:")
                    SigConEntryMinGain.place(relx=0.45, rely=0.45, anchor="center")
                    SigConEntryMinGain.config(state="normal")
                    SigConEntryMaxGain.place(relx=0.67, rely=0.45, anchor="center")  # Red max entry
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

                    LogConLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    LogConLabelMin.config(text="Minimum:")
                    LogConLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    LogConLabelMax.config(text="Maximum:")
                    LogConEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    LogConEntryMin.config(state="normal")
                    LogConEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
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

                    LinConLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    LinConLabelMin.config(text="Minimum:")
                    LinConLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    LinConLabelMax.config(text="Maximum:")
                    LinConEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    LinConEntryMin.config(state="normal")
                    LinConEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
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

                    CLAHELabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    CLAHELabelMin.config(text="Minimum:")
                    CLAHELabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    CLAHELabelMax.config(text="Maximum:")
                    CLAHEEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    CLAHEEntryMin.config(state="normal")
                    CLAHEEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
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

                    SharpLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    SharpLabelMin.config(text="Minimum:")
                    SharpLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    SharpLabelMax.config(text="Maximum:")
                    SharpEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    SharpEntryMin.config(state="normal")
                    SharpEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
                    SharpEntryMax.config(state="normal")

                    AlphaLabel.place(relx=0.5, rely=0.35, anchor="center")

                    AlphaLabelMin.place(relx=0.35, rely=0.45, anchor="center")
                    AlphaLabelMin.config(text="Minimum:")
                    AlphaLabelMax.place(relx=0.57, rely=0.45, anchor="center")
                    AlphaLabelMax.config(text="Maximum:")
                    AlphaEntryMin.place(relx=0.45, rely=0.45, anchor="center")
                    AlphaEntryMin.config(state="normal")
                    AlphaEntryMax.place(relx=0.67, rely=0.45, anchor="center")  # Red max entry
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

                    EmbLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    EmbLabelMin.config(text="Minimum:")
                    EmbLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    EmbLabelMax.config(text="Maximum:")
                    EmbEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    EmbEntryMin.config(state="normal")
                    EmbEntryMax.place(relx=0.67, rely=0.27, anchor="center")  # Red max entry
                    EmbEntryMax.config(state="normal")

                    AlphaEmbLabel.place(relx=0.5, rely=0.38, anchor="center")

                    AlphaEmbLabelMin.place(relx=0.35, rely=0.45, anchor="center")
                    AlphaEmbLabelMin.config(text="Minimum:")
                    AlphaEmbLabelMax.place(relx=0.57, rely=0.45, anchor="center")
                    AlphaEmbLabelMax.config(text="Maximum:")
                    AlphaEmbEntryMin.place(relx=0.45, rely=0.45, anchor="center")
                    AlphaEmbEntryMin.config(state="normal")
                    AlphaEmbEntryMax.place(relx=0.67, rely=0.45, anchor="center")  # Red max entry
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
            SPCoarseCheck = tk.Checkbutton(Frame, variable=SPCoarseVar, onvalue=1, offvalue=0, bg="gray86")

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

                    NoiseGaussLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    NoiseGaussLabelMin.config(text="Minimum:")
                    NoiseGaussLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    NoiseGaussLabelMax.config(text="Maximum:")
                    NoiseGaussEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    NoiseGaussEntryMin.config(state="normal")
                    NoiseGaussEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    NoiseLapLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    NoiseLapLabelMin.config(text="Minimum:")
                    NoiseLapLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    NoiseLapLabelMax.config(text="Maximum:")
                    NoiseLapEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    NoiseLapEntryMin.config(state="normal")
                    NoiseLapEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    NoisePoiLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    NoisePoiLabelMin.config(text="Minimum:")
                    NoisePoiLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    NoisePoiLabelMax.config(text="Maximum:")
                    NoisePoiEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    NoisePoiEntryMin.config(state="normal")
                    NoisePoiEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    SPLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    SPLabelMin.config(text="Minimum:")
                    SPLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    SPLabelMax.config(text="Maximum:")
                    SPEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    SPEntryMin.config(state="normal")
                    SPEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    NoiseImpLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    NoiseImpLabelMin.config(text="Minimum:")
                    NoiseImpLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    NoiseImpLabelMax.config(text="Maximum:")
                    NoiseImpEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    NoiseImpEntryMin.config(state="normal")
                    NoiseImpEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    JPEGLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    JPEGLabelMin.config(text="Minimum:")
                    JPEGLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    JPEGLabelMax.config(text="Maximum:")
                    JPEGEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    JPEGEntryMin.config(state="normal")
                    JPEGEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    SolLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    SolLabelMin.config(text="Minimum:")
                    SolLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    SolLabelMax.config(text="Maximum:")
                    SolEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    SolEntryMin.config(state="normal")
                    SolEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    ShotLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    ShotLabelMin.config(text="Minimum:")
                    ShotLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    ShotLabelMax.config(text="Maximum:")
                    ShotEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    ShotEntryMin.config(state="normal")
                    ShotEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    SpecLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    SpecLabelMin.config(text="Minimum:")
                    SpecLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    SpecLabelMax.config(text="Maximum:")
                    SpecEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    SpecEntryMin.config(state="normal")
                    SpecEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

            CorDropLabel = tk.Label(Frame, text="Enter relative size percentage. \n Use values between 0 and 1")

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

                    CutoutLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    CutoutLabelMin.config(text="Minimum:")
                    CutoutLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    CutoutLabelMax.config(text="Maximum:")
                    CutoutEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    CutoutEntryMin.config(state="normal")
                    CutoutEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    DropoutLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    DropoutLabelMin.config(text="Minimum:")
                    DropoutLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    DropoutLabelMax.config(text="Maximum:")
                    DropoutEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    DropoutEntryMin.config(state="normal")
                    DropoutEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    CorDropoutLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    CorDropoutLabelMin.config(text="Minimum:")
                    CorDropoutLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    CorDropoutLabelMax.config(text="Maximum:")
                    CorDropoutEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    CorDropoutEntryMin.config(state="normal")
                    CorDropoutEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    CorDropoutEntryMax.config(state="normal")
                    CorDropLabel.place(relx=0.5, rely=0.35, anchor="center")

                    CorDropoutSizeLabelMin.place(relx=0.35, rely=0.45, anchor="center")
                    CorDropoutSizeLabelMin.config(text="Width:")
                    CorDropoutSizeLabelMax.place(relx=0.57, rely=0.45, anchor="center")
                    CorDropoutSizeLabelMax.config(text="Height:")
                    CorDropoutSizeEntryMin.place(relx=0.45, rely=0.45, anchor="center")
                    CorDropoutSizeEntryMin.config(state="normal")
                    CorDropoutSizeEntryMax.place(relx=0.67, rely=0.45, anchor="center")
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

                    GaussBlurLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    GaussBlurLabelMin.config(text="Minimum:")
                    GaussBlurLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    GaussBlurLabelMax.config(text="Maximum:")
                    GaussBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    GaussBlurEntryMin.config(state="normal")
                    GaussBlurEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    AvgBlurLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    AvgBlurLabelMin.config(text="Minimum:")
                    AvgBlurLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    AvgBlurLabelMax.config(text="Maximum:")
                    AvgBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    AvgBlurEntryMin.config(state="normal")
                    AvgBlurEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    MedBlurLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    MedBlurLabelMin.config(text="Minimum:")
                    MedBlurLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    MedBlurLabelMax.config(text="Maximum:")
                    MedBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    MedBlurEntryMin.config(state="normal")
                    MedBlurEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    BiBlurLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    BiBlurLabelMin.config(text="Minimum:")
                    BiBlurLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    BiBlurLabelMax.config(text="Maximum:")
                    BiBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    BiBlurEntryMin.config(state="normal")
                    BiBlurEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    GlassBlurLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    GlassBlurLabelMin.config(text="Minimum:")
                    GlassBlurLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    GlassBlurLabelMax.config(text="Maximum:")
                    GlassBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    GlassBlurEntryMin.config(state="normal")
                    GlassBlurEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    MotBlurLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    MotBlurLabelMin.config(text="Minimum:")
                    MotBlurLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    MotBlurLabelMax.config(text="Maximum:")
                    MotBlurEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    MotBlurEntryMin.config(state="normal")
                    MotBlurEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    MotBlurEntryMax.config(state="normal")

                    MotBlurDirLabel = tk.Label(Frame, text="Enter a range of directions for blur (0-360)", bg="gray86")
                    MotBlurDirLabel.place(relx=0.5, rely=0.35, anchor="center")

                    MotBlurDirLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    MotBlurDirLabelMin.config(text="Minimum:")
                    MotBlurDirLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    MotBlurDirLabelMax.config(text="Maximum:")
                    MotBlurDirEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    MotBlurDirEntryMin.config(state="normal")
                    MotBlurDirEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    XMin = float(XscaleEntryMin.get())
                    XMax = float(XscaleEntryMax.get())
                    YMin = float(YscaleEntryMin.get())
                    YMax = float(YscaleEntryMax.get())

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
            # TODO Add checkbox for y shear ans x shear
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

                    XscaleLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    XscaleLabelMin.config(text="Minimum X:")
                    XscaleLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    XscaleLabelMax.config(text="Maximum X:")
                    XscaleEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    XscaleEntryMin.config(state="normal")
                    XscaleEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    XscaleEntryMax.config(state="normal")

                    YscaleLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    YscaleLabelMin.config(text="Minimum Y:")
                    YscaleLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    YscaleLabelMax.config(text="Maximum Y:")
                    YscaleEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    YscaleEntryMin.config(state="normal")
                    YscaleEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    TransXLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    TransXLabelMin.config(text="Minimum X:")
                    TransXLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    TransXLabelMax.config(text="Maximum X:")
                    TransXEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    TransXEntryMin.config(state="normal")
                    TransXEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    TransXEntryMax.config(state="normal")

                    TransYLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    TransYLabelMin.config(text="Minimum Y:")
                    TransYLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    TransYLabelMax.config(text="Maximum Y:")
                    TransYEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    TransYEntryMin.config(state="normal")
                    TransYEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    RotLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    RotLabelMin.config(text="Minimum:")
                    RotLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    RotLabelMax.config(text="Maximum:")
                    RotEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    RotEntryMin.config(state="normal")
                    RotEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    XshearLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    XshearLabelMin.config(text="Minimum X:")
                    XshearLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    XshearLabelMax.config(text="Maximum X:")
                    XshearEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    XshearEntryMin.config(state="normal")
                    XshearEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    XshearEntryMax.config(state="normal")

                    YshearLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    YshearLabelMin.config(text="Minimum Y:")
                    YshearLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    YshearLabelMax.config(text="Maximum Y:")
                    YshearEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    YshearEntryMin.config(state="normal")
                    YshearEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    PALabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    PALabelMin.config(text="Minimum:")
                    PALabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    PALabelMax.config(text="Maximum:")
                    PAEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    PAEntryMin.config(state="normal")
                    PAEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    ETLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    ETLabelMin.config(text="Minimum:")
                    ETLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    ETLabelMax.config(text="Maximum:")
                    ETLabelSigma.place(relx=0.5, rely=0.35, anchor="center")
                    ETLabelSigma.config(text="Sigma: enter values between 0 and 1")
                    ETEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    ETEntryMin.config(state="normal")
                    ETEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    polarLabelMin.place(relx=0.35, rely=0.4, anchor="center")
                    polarLabelMin.config(text="Minimum:")
                    polarLabelMax.place(relx=0.57, rely=0.4, anchor="center")
                    polarLabelMax.config(text="Maximum:")
                    polarEntryMin.place(relx=0.45, rely=0.4, anchor="center")
                    polarEntryMin.config(state="normal")
                    polarEntryMax.place(relx=0.67, rely=0.4, anchor="center")
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

                    JigRowsLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    JigRowsLabelMin.config(text="Rows Min:")
                    JigRowsLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    JigRowsLabelMax.config(text="Rows Max:")
                    JigRowsEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    JigRowsEntryMin.config(state="normal")
                    JigRowsEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    JigRowsEntryMax.config(state="normal")

                    JigColsLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    JigColsLabelMin.config(text="Columns Min:")
                    JigColsLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    JigColsLabelMax.config(text="Columns Max:")
                    JigColsEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    JigColsEntryMin.config(state="normal")
                    JigColsEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    CannyLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    CannyLabelMin.config(text="Minimum:")
                    CannyLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    CannyLabelMax.config(text="Maximum")
                    CannyEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    CannyEntryMin.config(state="normal")
                    CannyEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    DirEdgeLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    DirEdgeLabelMin.config(text="Minimum:")
                    DirEdgeLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    DirEdgeLabelMax.config(text="Maximum:")
                    DirEdgeEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    DirEdgeEntryMin.config(state="normal")
                    DirEdgeEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    DirEdgeEntryMax.config(state="normal")

                    DirEdgeAlphaLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    DirEdgeAlphaLabelMin.config(text="Minimum:")
                    DirEdgeAlphaLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    DirEdgeAlphaLabelMax.config(text="Maximum:")
                    DirEdgeAlphaEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    DirEdgeAlphaEntryMin.config(state="normal")
                    DirEdgeAlphaEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    SupixLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    SupixLabelMin.config(text="Minimum:")
                    SupixLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    SupixLabelMax.config(text="Maximum:")
                    SupixEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    SupixEntryMin.config(state="normal")
                    SupixEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    VoRowsLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    VoRowsLabelMin.config(text="Rows Min:")
                    VoRowsLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    VoRowsLabelMax.config(text="Rows Max:")
                    VoRowsEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    VoRowsEntryMin.config(state="normal")
                    VoRowsEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    VoRowsEntryMax.config(state="normal")

                    VoColsLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    VoColsLabelMin.config(text="Cols Min:")
                    VoColsLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    VoColsLabelMax.config(text="Cols Max:")
                    VoColsEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    VoColsEntryMin.config(state="normal")
                    VoColsEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    UniVoRowsLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    UniVoRowsLabelMin.config(text="Minimum:")
                    UniVoRowsLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    UniVoRowsLabelMax.config(text="Maximum:")
                    UniVoRowsEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    UniVoRowsEntryMin.config(state="normal")
                    UniVoRowsEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

                    RegRowsLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    RegRowsLabelMin.config(text="Rows Min:")
                    RegRowsLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    RegRowsLabelMax.config(text="Rows Max:")
                    RegRowsEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    RegRowsEntryMin.config(state="normal")
                    RegRowsEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    RegRowsEntryMax.config(state="normal")

                    RegColsLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    RegColsLabelMin.config(text="Cols Min:")
                    RegColsLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    RegColsLabelMax.config(text="Cols Max:")
                    RegColsEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    RegColsEntryMin.config(state="normal")
                    RegColsEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    SnowsizeLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    SnowsizeLabelMin.config(text="Size Min:")
                    SnowsizeLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    SnowsizeLabelMax.config(text="Size Max:")
                    SnowsizeEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    SnowsizeEntryMin.config(state="normal")
                    SnowsizeEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    SnowsizeEntryMax.config(state="normal")

                    SnowspeedLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    SnowspeedLabelMin.config(text="Speed Min:")
                    SnowspeedLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    SnowspeedLabelMax.config(text="Speed Max:")
                    SnowspeedEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    SnowspeedEntryMin.config(state="normal")
                    SnowspeedEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    RainsizeLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    RainsizeLabelMin.config(text="Size Min:")
                    RainsizeLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    RainsizeLabelMax.config(text="Size Max:")
                    RainsizeEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    RainsizeEntryMin.config(state="normal")
                    RainsizeEntryMax.place(relx=0.67, rely=0.27, anchor="center")
                    RainsizeEntryMax.config(state="normal")

                    RainspeedLabelMin.place(relx=0.35, rely=0.43, anchor="center")
                    RainspeedLabelMin.config(text="Speed Min:")
                    RainspeedLabelMax.place(relx=0.57, rely=0.43, anchor="center")
                    RainspeedLabelMax.config(text="Speed Max:")
                    RainspeedEntryMin.place(relx=0.45, rely=0.43, anchor="center")
                    RainspeedEntryMin.config(state="normal")
                    RainspeedEntryMax.place(relx=0.67, rely=0.43, anchor="center")
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

                    FrostLabelMin.place(relx=0.35, rely=0.27, anchor="center")
                    FrostLabelMin.config(text="Min:")
                    FrostLabelMax.place(relx=0.57, rely=0.27, anchor="center")
                    FrostLabelMax.config(text="Max:")
                    FrostEntryMin.place(relx=0.45, rely=0.27, anchor="center")
                    FrostEntryMin.config(state="normal")
                    FrostEntryMax.place(relx=0.67, rely=0.27, anchor="center")
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

        def Scrolly(e):  # Scroll through box if necessary
            AugCanvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def DisplayImages():
            OrigLabel = tk.Label(Aug_Interface, text="Original Images", font=20)

            OrigLabel.place(anchor="center", relx=0.7, rely=0.1)

            LastSetButton.place(anchor="center", relx=0.65, rely=0.63)
            NextSetButton.place(anchor="center", relx=0.75, rely=0.63)
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
                OrigCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
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
                OrigCanvas.create_image(CanvasXLoc, 110, image=Nigel, anchor="w")
                CanvasXLoc += (width + DeadspaceFraction)

            try:
                PreviewAugmentations()
            except:
                pass

        def ImageNumberPlacement(event):
            if OptionDropdown.get() == "Apply To Specific Number Of Images":
                NoImageLabel.place(anchor="center", relx=0.67, rely=0.76)
                NoImageEntry.place(anchor="center", relx=0.73, rely=0.76)
            else:
                try:
                    NoImageEntry.place_forget()
                    NoImageLabel.place_forget()
                except:
                    pass

        def NewSetPlacement():
            if SetVar.get() == 2:

                NewSetLabel2.place(anchor="center", relx=0.7, rely=0.83)
                NewSetEntry.place(anchor="center", relx=0.68, rely=0.86)
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
            NewSetLabel2.config(fg="black", text="Select your output folder")

            if len(Step1Entry.get()) == 0:
                Step1Text.config(fg="red")
                Error += 1

            if not os.path.isdir(str(Step1Entry.get())):
                Step1Text.config(fg="red", text="Not a valid folder")
                Error += 1

            if len(SavedOptionCommand) == 0:
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
                Update_pgbarB()

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

        def Update_pgbarB():
            nonlocal Processed_Images
            pgbar.config(maximum=len(ImagesForAug))
            while Processed_Images < len(ImagesForAug):
                BeginButton.config(state="disabled")
                ProgressVarA.set(Processed_Images)
                pgbarlablel.config(text=f"Images Processed : {Processed_Images}/{len(ImagesForAug)}")
                Aug_Interface.update()
                time.sleep(0.000001)

            pgbarlablel.config(text=f"Images Processed: {Processed_Images} / {len(ImagesForAug)}")
            ProgressVarA.set(Processed_Images)
            Aug_Interface.update()
            BeginButton.config(state="normal")
            Pop()

        def AugCore(ImagePath):
            nonlocal Processed_Images
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
                Processed_Images += 1

            elif SetVar.get() == 1:
                io.imsave(ImagePath, Image_Out)
                Processed_Images += 1

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
                Processed_Images += 1

            print(Processed_Images)

        def Pop():

            Popup = tk.Toplevel(HomeScreen)
            Popup.geometry(f"200x100+{ScreenWidthMiddle - 100}+{ScreenHeightMiddle - 50}")
            Popup.title("Complete")
            Popup.resizable(False, False)
            Popup.iconbitmap(r"Icon/HistoSquare.ico")
            Info = tk.Label(Popup, text="Process Complete\n"
                                        "Stay here or return to home screen?")

            Popup.grab_set()

            def ReturnHome():
                Popup.destroy()
                HomeScreen.deiconify()
                Aug_Interface.destroy()

            def Stay():
                Popup.destroy()
                Popup.grab_release()
                ResetAll()

            Info.place(relx=0.5, rely=0.2, anchor="center")
            Home_Button = tk.Button(Popup, text="Home", command=ReturnHome, width=10)
            Home_Button.place(relx=0.25, rely=0.6, anchor="center")
            Stay_Button = tk.Button(Popup, text="Stay", command=Stay, width=10)
            Stay_Button.place(relx=0.75, rely=0.6, anchor="center")

        def ResetAll():
            nonlocal Processed_Images
            Processed_Images = 0
            ProgressVarA.set(0)
            NewSetEntry.delete(0, tk.END)
            NoImageEntry.delete(0, tk.END)
            ProgressLabel.config(text="")
            pgbarlablel.config(text="")

        # Aug Interface Main Screen
        LargeFont = font.Font(family="Helvetica", size=15)
        LargeSkinnyFont = font.Font(size=13)
        Aug_Interface = tk.Toplevel(HomeScreen)
        Aug_Interface.title(f"Image Augmentation")
        Aug_Interface.geometry(f"1800x1000+{ScreenWidthMiddle - 900}+{0}")
        Aug_Interface.iconbitmap(r"Icon/HistoSquare.ico")
        Aug_Interface.resizable(False, False)
        Aug_Interface.protocol("WM_DELETE_WINDOW", lambda: Close_And_Home(Aug_Interface))

        InfoLabel = tk.Label(Aug_Interface, text=f"This module will allow you to augment your images", font=LargeFont)
        InfoLabel.place(anchor="center", relx=0.5, rely=0.02)

        Step1Text = tk.Label(Aug_Interface, fg="black", text="Step 1: Please select the images you want to augment:",
                             font=20)
        Step1Text.place(anchor="center", relx=0.175, rely=0.05)
        Step1Entry = tk.Entry(Aug_Interface, width=30, font=LargeSkinnyFont)
        Step1Entry.place(anchor="center", relx=0.16, rely=0.08)
        ButtonForFiles = tk.Button(Aug_Interface, text="Browse...", command=DisplayImages)
        ButtonForFiles.place(anchor="center", relx=0.255, rely=0.08)

        ImageNoLabel = tk.Label(Aug_Interface, text="", font=20, fg="blue")
        ImageNoLabel.place(anchor="center", relx=0.175, rely=0.12)

        AugInstructText = tk.Label(Aug_Interface, text="Step 2: Select your augmentations", font=20)
        AugInstructText.place(anchor="center", relx=0.175, rely=0.155)

        AugFrameOutter = tk.Frame(Aug_Interface, width=500, height=300, borderwidth=2, relief="ridge")
        AugFrameOutter.place(anchor="center", relx=0.175, rely=0.32)
        AugCanvas = tk.Canvas(AugFrameOutter, width=500, height=300, bg="gray86", scrollregion=(0, 0, 500, 300),
                              highlightthickness=0)
        AugScroll = tk.Scrollbar(AugFrameOutter, orient="vertical", command=AugCanvas.yview)
        AugScroll.pack(side="left", fill="y")
        AugCanvas.config(yscrollcommand=AugScroll.set)
        AugCanvas.pack(side="left", expand=True, fill="both")
        AugCanvas.bind_all("<MouseWheel>", Scrolly)

        OptionsFrame = tk.Frame(Aug_Interface, width=520, height=400, borderwidth=2, relief="ridge", bg="grey86")
        OptionsFrame.place(anchor="center", relx=0.175, rely=0.7)

        SubDescriptionLabel = tk.Label(OptionsFrame, text="", bg="gray86")

        OptionsText = tk.Label(OptionsFrame, text="Image Options Appear Here.", bg="grey86")
        OptionsText.place(anchor="center", relx=0.5, rely=0.5)

        AddOptionButton = tk.Button(Aug_Interface, text="Add", font=15, width=10, command=AddOption)
        AddOptionButton.place(anchor="center", relx=0.35, rely=0.32)

        RemoveOptionButton = tk.Button(Aug_Interface, text="Remove", font=15, width=10, command=RemoveOption)
        RemoveOptionButton.place(anchor="center", relx=0.35, rely=0.39)

        OrigCanvas = tk.Canvas(Aug_Interface, width=1000, height=250)
        OrigCanvas.place(anchor="center", relx=0.7, rely=0.25)

        PrevCanvas = tk.Canvas(Aug_Interface, width=1000, height=250)
        PrevCanvas.place(anchor="center", relx=0.7, rely=0.50)

        LastSetButton = tk.Button(Aug_Interface, text="Prev", width=10, command=PrevSet)

        NextSetButton = tk.Button(Aug_Interface, text="Next", width=10, command=NextSet)

        HowLabel = tk.Label(Aug_Interface, text="How do you want these changes applied?", font=20)
        HowLabel.place(anchor="center", relx=0.7, rely=0.7)

        OptionList2 = ["Apply To All Images", "Apply To Random Amount Of Images", "Apply To Specific Number Of Images"]
        OptionDropdown = ttk.Combobox(Aug_Interface, values=OptionList2, width=35, state="readonly")
        OptionDropdown.place(relx=0.7, rely=0.73, anchor="center")
        OptionDropdown.bind("<<ComboboxSelected>>", ImageNumberPlacement)

        NoImageLabel = tk.Label(Aug_Interface, text="Number of Images")
        NoImageEntry = tk.Entry(Aug_Interface, width=7)

        SetVar = tk.IntVar()
        SetVar.set(1)
        SameSetRadio = tk.Radiobutton(Aug_Interface, text="Replace Selected Images", variable=SetVar, value=1,
                                      command=NewSetPlacement)
        SameSetRadio.place(anchor="center", relx=0.6, rely=0.8)
        NewSetRadio = tk.Radiobutton(Aug_Interface, text="Create New Image Set", variable=SetVar, value=2,
                                     command=NewSetPlacement)
        NewSetRadio.place(anchor="center", relx=0.7, rely=0.8)

        AddSetRadio = tk.Radiobutton(Aug_Interface, text="Add to Selected Images", variable=SetVar, value=3,
                                     command=NewSetPlacement)
        AddSetRadio.place(anchor="center", relx=0.8, rely=0.8)

        NewSetLabel2 = tk.Label(Aug_Interface, text="Select your output folder")
        NewSetEntry = tk.Entry(Aug_Interface, width=40)
        NewSetBrowse = tk.Button(Aug_Interface, text="Browse...", command=GetNewSetLoc)

        ProgressLabel = tk.Label(Aug_Interface, fg="black", text="")
        ProgressLabel.place(anchor="center", relx=0.5, rely=0.93)

        BeginButton = tk.Button(Aug_Interface, text="Start", width=10, command=Aug_Thread_Threading)
        BeginButton.place(anchor="center", relx=0.67, rely=0.9)

        BackButton = tk.Button(Aug_Interface, text="Back", width=10)
        BackButton.place(anchor="center", relx=0.72, rely=0.9)

        pgbarlablel = tk.Label(Aug_Interface, text="")
        pgbarlablel.place(anchor="center", relx=0.7, rely=0.95)

        ProgressVarA = tk.IntVar()
        pgbar = ttk.Progressbar(Aug_Interface, length=1000, orient="horizontal", maximum=100, value=0,
                                variable=ProgressVarA)
        pgbar.place(anchor="center", relx=0.7, rely=0.97)

        ################################# Options Widets ###############################################

        DescLabel = tk.Label(OptionsFrame, text="", bg="grey86")
        DescLabel.place(relx=0.5, rely=0.05, anchor="center")

        tk.mainloop()

def call_mainroot():
    SplashRoot.destroy()
    mainroot()

SplashRoot.after(4000, call_mainroot)
tk.mainloop()
