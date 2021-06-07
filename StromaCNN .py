import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as ff
import torch.optim as optim
import csv
from matplotlib import style
from datetime import datetime
import pandas as pd
from sklearn.metrics import roc_curve, auc
import shutil
from shutil import copyfile
import playsound

Date = datetime.now() # Get Date and Time.  This will be used in naming each of our runs
DateString = Date.strftime('%Y-%m-%d   %H-%M-%S') # Convert the date and time into a readable string

############################################# User Interactive Variables ###############################################

Description = ''
Model_Name = '' # The name of the model, change the string to customise

REBUILD_DATA = True # Do you want to add the images again?
Run_On_GPU = True # Would you like to run on a GPU (Where available)
Load_Pretrained_Model = False # Would you like to run a pre-trained model? If True, add file path to "Pre-trained_Model_Filepath
Train_Model = True # Do you want to train the model?
Pretrained_Model_Filepath = r""  # Filepath to pre-trained model where applicable

Image_Size = 250 # Select the image size in pixels. This is largely based on the memory available in the GPU

Mature_Training_Folder = r""
Immature_Training_Folder = r""
Mature_Test_Folder = r""
Immature_Test_Folder = r""


Input_Nodes = 1
Conv_Kernal_Size = 5
Pool_Kernal_Size = 2
First_Layer_Nodes = 32
Second_Layer_Nodes = 64
Third_Layer_Nodes = 128
Fourth_Layer_Nodes = 128
Fifth_Layer_Nodes = 128
First_Linear_Layer_Nodes = 512
Final_Layer_Nodes = 2
Stride = 1
Learning_Rate = 0.00001
Batch_Size = 150
Training_Epochs = 200
Restart_Epoch = 0
Test_Test_Rate = 1

# Decide thresholds for training module to be saved and independently validated.
Training_acc_thresh = 0.75
Test_acc_thresh = 0.75

Validation_Batch_Size = 150

############################################## Create Folder for Results ###############################################

Results_Folder = Model_Name # Define the name of the results folder.  Here we just use the model name
try:
    os.mkdir(Model_Name) # Make the results folder
    os.mkdir(Model_Name + "\\Model savestates") # Make a folder in the results folder to save the best models
except:
    pass



################################################ Check GPU Availability ################################################

if torch.cuda.is_available():
    if Run_On_GPU:
        device = torch.device("cuda:0")
        print("Running Network on GPU")
    else:
        device = torch.device("cpu")
        print("Running Network on CPU")
else:
    device = torch.device("cpu")
    print("Running Network on CPU")

####################################### Fetch the Images and prepare them for analysis #################################

class DataLoader:
    Image_Size = Image_Size # Bring the image size into the local class

    # Provide the locations of your image folders
    Mature_Training = Mature_Training_Folder
    Immature_Training = Immature_Training_Folder
    Mature_Test = Mature_Test_Folder
    Immature_Test = Immature_Test_Folder
    Mature_Independent = Mature_Validation_Folder
    Immature_Independent = Immature_Validation_Folder

    # Label the Folders in binary; 0 = negative, 1 = positive
    Labels_Training = {Mature_Training: 0, Immature_Training: 1}
    Labels_Validation = {Mature_Test: 0, Immature_Test: 1}
    Labels_Independent = {Mature_Independent: 0, Immature_Independent: 1}

    # Initiate empty plits for loading the data
    Training_Data = []
    Image_Paths_Training = []
    Validation_Data = []
    Image_Paths_Validation = []
    Independent_Data = []
    Independent_Image_Paths = []

    # Initate zero counts for counting the images coming through
    Immature_Count_Training = 0
    Mature_Count_Training = 0
    Image_Count_Training = 0
    Immature_Count_Test = 0
    Mature_Count_Test = 0
    Image_Count_Test = 0
    Immature_Count_Independent = 0
    Mature_Count_Independent = 0
    Image_Count_Independent = 0

    # This function makes the training data
    def Make_Training_Data(self):
        print("Loading Training Images")
        for label in self.Labels_Training: # For each of the folders in Labels_Training...
            # print(label)
            for o in tqdm(os.listdir(label)): # For the +ve and -ve folders...
                for g in os.listdir(label + "\\" + o): # For the patient folders
                    for k in os.listdir(label + "\\" + o + "\\" + g):

                        try: # Try to get the image
                            path = os.path.join(label + "\\" + o + "\\" + g + "\\" + k)
                            #print(path)
                            if "jpg" in path:

                                pathID = self.Image_Count_Training # Give an ID number to the image based off Image_Count_Training
                                self.Image_Count_Training += 1 # Add one to the image count
                                self.Image_Paths_Training.append(path) # Append the image filepath to the Image_Paths_Training list
                                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # use the cv2 package to read in the image as a greyscale image
                                img = cv2.resize(img, (self.Image_Size, self.Image_Size)) # Resize the image to the pixel value set in Image_Size
                                self.Training_Data.append(
                                    [np.array(img), np.eye(2)[self.Labels_Training[label]], [path], [pathID]])
                                # The above line creates a list of detais about the image, [pixel values, ground truth, filepath to the image, numeric ID]

                                if label == self.Mature_Training:
                                    self.Mature_Count_Training += 1 # Add 1 to the Negative count if image is negative
                                elif label == self.Immature_Training:
                                    self.Immature_Count_Training += 1 # Add 1 to the positive count if image is positive
                        except Exception as e:
                            print(str(e))
                            pass

        np.random.shuffle(self.Training_Data)
        np.save(Results_Folder + "\\Training_Data.npy", self.Training_Data)
        print("Immature Training Images: ", self.Immature_Count_Training)
        print("Mature Training Images: ", self.Mature_Count_Training)

    def Make_Test_Data(self):
        print("Loading Validation Images")
        for label in self.Labels_Validation:
            for o in tqdm(os.listdir(label)):
                for g in os.listdir(label + "\\" + o):
                    for k in os.listdir(label + "\\" + o + "\\" + g):

                        try:
                            path = os.path.join(label + "\\" + o + "\\" + g + "\\" + k)
                            if "jpg" in path:

                                pathID = self.Image_Count_Test
                                self.Image_Count_Test += 1
                                self.Image_Paths_Validation.append(path)
                                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                img = cv2.resize(img, (self.Image_Size, self.Image_Size))
                                self.Validation_Data.append(
                                    [np.array(img), np.eye(2)[self.Labels_Validation[label]], [path], [pathID]])

                                if label == self.Mature_Test:
                                    self.Mature_Count_Test += 1
                                elif label == self.Immature_Test:
                                    self.Immature_Count_Test += 1
                        except Exception as e:
                            print(str(e))
                            pass

        np.random.shuffle(self.Validation_Data)
        np.save(Results_Folder + "\\Validation_Data.npy", self.Validation_Data)
        print("Immature Validation Images: ", self.Immature_Count_Test)
        print("Mature Validation Images: ", self.Mature_Count_Test)



#################################################### Make the Model ####################################################

class StromaDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(Input_Nodes, First_Layer_Nodes, Conv_Kernal_Size, Stride)  # Layer 1
        self.conv2 = nn.Conv2d(First_Layer_Nodes, Second_Layer_Nodes, Conv_Kernal_Size, Stride)  # Layer 2
        self.conv3 = nn.Conv2d(Second_Layer_Nodes, Third_Layer_Nodes, Conv_Kernal_Size, Stride)  # Layer 3
        self.conv4 = nn.Conv2d(Third_Layer_Nodes, Fourth_Layer_Nodes, Conv_Kernal_Size, Stride)  # Layer 4
        self.conv5 = nn.Conv2d(Fourth_Layer_Nodes, Fifth_Layer_Nodes, Conv_Kernal_Size, Stride)  # Layer 5


        # This bit creates a sample image and sends it through to get the correct inputs for the flattened layer
        Sample_Image_Array = torch.randn(Image_Size, Image_Size).view(-1, 1, Image_Size, Image_Size)
        self.convs(Sample_Image_Array)
        self.fc1 = nn.Linear(self._to_linear, First_Linear_Layer_Nodes)
        self.fc2 = nn.Linear(First_Linear_Layer_Nodes, Final_Layer_Nodes)

    def convs(self, x):
        x = ff.avg_pool2d(ff.relu(self.conv1(x)), (Pool_Kernal_Size, Pool_Kernal_Size))
        x = ff.avg_pool2d(ff.relu(self.conv2(x)), (Pool_Kernal_Size, Pool_Kernal_Size))
        x = ff.avg_pool2d(ff.relu(self.conv3(x)), (Pool_Kernal_Size, Pool_Kernal_Size))
        x = ff.avg_pool2d(ff.relu(self.conv4(x)), (Pool_Kernal_Size, Pool_Kernal_Size))
        x = ff.avg_pool2d(ff.relu(self.conv5(x)), (Pool_Kernal_Size, Pool_Kernal_Size))

        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = ff.relu(self.fc1(x))
        x = self.fc2(x)
        return ff.softmax(x, dim=1)


Stroma_CNN = StromaDetector().to(device) # Move the model onto the GPU
optimiser = optim.Adam(Stroma_CNN.parameters(), lr=Learning_Rate) # Define the optimiser
loss_function = nn.MSELoss() # Define the loss function

############################################ Make the image tensors ####################################################

# Use the created DataLoader() Class to make the data form the images
if REBUILD_DATA:
    DataLoader().Make_Training_Data()
    DataLoader().Make_Test_Data()


# Load up the created training data
training_data = np.load(Results_Folder + "\\Training_Data.npy", allow_pickle=True)
val_data = np.load(Results_Folder + "\\Validation_Data.npy", allow_pickle=True)

print("Total Training images: ", len(training_data))
print("Total Validation images: ", len(val_data))

print("Translating image data to tensors")  # Getting the images the right size, greyscaling and normalising between 0 and 1
Train_X = torch.Tensor([i[0] for i in training_data]).view(-1, Image_Size, Image_Size)  # Image arrays
Train_X = Train_X / 255.0

Train_Ids = torch.Tensor([i[1] for i in training_data])

Train_Paths = [i[2] for i in training_data]

Val_X = torch.Tensor([i[0] for i in val_data]).view(-1, Image_Size, Image_Size)  # Image arrays
Val_X = Val_X / 255.0
Val_Ids = torch.Tensor([i[1] for i in val_data])
Val_Paths = [i[2] for i in val_data]

######################################## Make functions for outcomes from our model ####################################

# Calculate the accuracy and the loss using the model
def Calculate_Acc_Loss(X, Y, train=False):
    if train:
        Stroma_CNN.zero_grad() # Resets the gradients so they don't accumulate between epochs
    outputs = Stroma_CNN(X) # Get the probabilities of each class from the model
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, Y)] # Check if the class with the biggest prob matches the ground truth
    acc = matches.count(True) / len(matches) # Get the accuracy
    loss = loss_function(outputs, Y) # Get the loss

    # Of we're training, use back propagation to adjust the weights and biases
    if train:
        loss.backward()
        optimiser.step()
    return acc, loss

# Make a function to litmus test our training periodically
def ValTest(size=150):
    random_start = np.random.randint(len(Val_X) - size)
    X, Y = Val_X[random_start:random_start + size], Val_Ids[random_start:random_start + size]
    with torch.no_grad():
        test_acc, test_loss = Calculate_Acc_Loss(X.view(-1, 1, Image_Size, Image_Size).to(device), Y.to(device))
        return test_acc, test_loss


###################################################### Train our model #################################################

def Train(net):
    print("Training the Model...")
    with open(Results_Folder + "\\model.log", "a")as f:  # "a" means append

        for epoch in range(Training_Epochs):
            ActEpoch = epoch+Restart_Epoch

            Batchy_Batch = 0
            print("Training epoch", epoch + 1, "of,", Training_Epochs)
            print(DateString)
            for i in tqdm(range(0, len(Train_X),
                                Batch_Size)):  # from 0, to the len of Training_X, stepping Batch_Size at a time.

                Batch_X = Train_X[i:i + Batch_Size].view(-1, 1, Image_Size, Image_Size).to(
                    device)  # These are the images of the batch flattened and stored on processor


                Batch_Y = Train_Ids[i:i + Batch_Size].to(
                    device)  # These are the ground truths of the batch flattened and stored on processor

                Training_acc, Training_loss = Calculate_Acc_Loss(Batch_X, Batch_Y, train=True)

                if (i / Batch_Size) % Test_Test_Rate == 0:
                    Val_Acc, Val_Loss = ValTest(size=Validation_Batch_Size)
                    f.write(
                        f"{Model_Name}, {epoch},{Batchy_Batch}, in_sample, {float(Training_acc)},{round(float(Training_loss), 5)}, Out_Of_Sample, {float(Val_Acc)}, {round(float(Val_Loss), 5)}\n")
                # Here we write all our data to the log file.  the \n means new line

                net.zero_grad()  # Resets the gradient we get from each batch to zero before we put the next one in. Reccomended to always do this.
                # The gradient contains the loss


                if Val_Acc >= Test_acc_thresh and Training_acc >= Training_acc_thresh:
                    torch.save(Stroma_CNN.state_dict(),
                               Model_Name + "\\Model savestates\\Epoch " + str(ActEpoch) + "Batch " + str(
                                   Batchy_Batch) + ' save.pt')
                Batchy_Batch += 1
            print(f"Loss:{Training_loss}")
            print("In Sample Accuracy:", round(Training_acc, 3))
            print("Out of Sample Accuracy:", round(Val_Acc, 3))

        print("Training complete!")

###################################### Define Function For Testing after training #################################################

Ground_Truth = []
Prediction_Percentage_Pos = []
Prediction_Percentage_Neg = []
Prediction_Class = []
Image_path = []
Correct = []


def Test(net):
    print("Testing model...")

    # Counters for correct and total
    Correct = 0
    Total = 0

    with torch.no_grad():
        for i in tqdm(range(len(Val_X))):  # Iterate over all the test images

            Image_Path = Val_Paths[i]  # Get the path for the current image
            Image_path.append(Image_Path)  # Add to empty list
            # print("Image:", Image_Path)
            # plt.imshow(Test_X[i])
            # plt.show()

            Real_Class = torch.argmax(Val_Ids[i]).to(
                device).item()  # The real class is the ground truth for these images. argmax gets the position of the largest value
            Ground_Truth.append(Real_Class)  # Add value to ground truth list
            # print("Actual Class:", Real_Class)

            Test_Out = net(Val_X[i].view(-1, 1, Image_Size, Image_Size).to(device))[
                0]  # Runs the image through the trained network. Provides proability prediction for each class
            Test_Out_Pos = Test_Out[1].item()  # Turn positive probability from tensor to python number
            Test_Out_Neg = Test_Out[0].item()  # Turn negative probability from tensor to python number

            Prediction_Percentage_Pos.append(
                Test_Out_Pos)  # Add these values to the positive and negative probability list
            Prediction_Percentage_Neg.append(Test_Out_Neg)
            # print("Prediction:", Test_Out)

            Test_Prediction = torch.argmax(Test_Out).item()  # Gets the position of the maximum probability
            Prediction_Class.append(Test_Prediction)  # Add to class list
            # print("Prediction:" ,Test_Prediction)

            if Test_Prediction == Real_Class:  # If the index of the max probability matches the index of the Ground truth...
                # print("Correct")
                Correct += 1
            # else:
            # print("Wrong")
            Total += 1

    print("Accuracy =", round(Correct / Total, 3))  # Get accuracy
    print("Correct:", Correct)  # Get correct stats
    print("Total:", Total)
    print("Testing Complete")
    Final_Acc = round(Correct / Total, 5)
    return Correct, Total, Final_Acc

################################################ Function for training ###############################################

if Load_Pretrained_Model: # If we have a pretrained model...
    try:
        Stroma_CNN.load_state_dict(torch.load(Pretrained_Model_Filepath))
        if Train_Model:
            Train(Stroma_CNN)
            Correct, Total, Final_Acc = Test(Stroma_CNN)
        else:
            Correct, Total, Final_Acc = Test(Stroma_CNN)
    except:
        print("ERROR, pretrained image dimensions different from input images")
else: # ...else we train the model form scratch.
    Train(Stroma_CNN)  # Run the training method
    Correct, Total, Final_Acc = Test(Stroma_CNN)  # This line actually runs the test method

########################################### Write results to CSV file ################################################

print("Writing results to CSV file")

Headers = ['Image', 'Ground Truth (1:Immature 0:Mature)', 'Immature Chance', 'Mature Chance',
           'Final Prediction']  # Titles for columns

with open((Results_Folder + "\\" + "Results.csv"), "w",
          newline='') as f:  # Creates the csv file open(name, write, new line delimiter)
    writer = csv.writer(f, delimiter=',')  # Shows what seperates values and what file to write to
    rows = zip(Image_path, Ground_Truth, Prediction_Percentage_Pos, Prediction_Percentage_Neg,
               Prediction_Class)  # Combine lists to table form
    writer.writerow(Headers)  # Write headers to csv file
    for row in rows:
        writer.writerow(row)  # Write data to csv file

print("Analysis Complete")

########################################### Create the Accuracy and loss graphs #################################################

style.use("ggplot")

model_name = Model_Name

style.use("ggplot")


def Create_Acc_Loss_Graph(mod):
    print("Creating Acc-Loss Graph")
    contents = open(Results_Folder + "\\model.log", "r").read().split("\n")  # The \n means split by new line

    Batch_List = []
    In_Acc_List = []
    In_Loss_List = []
    Out_Acc_List = []
    Out_Loss_List = []
    Epoch_List = []

    for c in contents:
        if model_name in c:
            Name, Epoch, Batch, Sample_type, In_Acc, In_Loss, Sample_type_two, Out_Acc, Out_Loss = c.split(",")

            In_Acc_List.append(In_Acc)
            In_Acc_List = [float(i) for i in In_Acc_List]
            In_Loss_List.append(In_Loss)
            In_Loss_List = [float(i) for i in In_Loss_List]
            Out_Acc_List.append(Out_Acc)
            Out_Acc_List = [float(i) for i in Out_Acc_List]
            Out_Loss_List.append(Out_Loss)
            Out_Loss_List = [float(i) for i in Out_Loss_List]
            Batch_List.append(Batch)
            Batch_List = [float(i) for i in Batch_List]
            Epoch_List.append(Epoch)
            Epoch_List = [float(i) for i in Epoch_List]

    Batch_List_Max = max(Batch_List)
    # print(Batch_List_Max)
    Batch_Fraction_List = [round(x / Batch_List_Max, 2) for x in Batch_List]
    # print(Batch_Fraction_List)
    Epoch_Fraction_List = [round(x + y, 2) for x, y in zip(Epoch_List, Batch_Fraction_List)]
    # print(Epoch_Fraction_List)

    fig = plt.figure(figsize=(19.20, 10.80))

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(Epoch_Fraction_List, In_Acc_List, label="Training Accuracy")
    ax1.plot(Epoch_Fraction_List, Out_Acc_List, label="Test Accuracy")
    ax1.legend(loc=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Training vs Test")

    ax2.plot(Epoch_Fraction_List, In_Loss_List, label="Training Loss")
    ax2.plot(Epoch_Fraction_List, Out_Loss_List, label="Test Loss")
    ax2.legend(loc=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Training vs Test")
    plt.savefig(Results_Folder + "\\Accuracy-Loss Graph")
    # plt.show()


if not Load_Pretrained_Model:
    Create_Acc_Loss_Graph(model_name)

if Load_Pretrained_Model:
    if Train_Model:
        Create_Acc_Loss_Graph(model_name)

############################################## Make Confusion Matrix ###################################################

plt.clf()


def Confusion_Matrix(CSV_Filepath):
    print("Generating Confusion Matrix")
    CSV = pd.DataFrame(pd.read_csv(CSV_Filepath, usecols=['Ground Truth (1:Immature 0:Mature)', 'Final Prediction']))
    Truth = CSV['Ground Truth (1:Immature 0:Mature)'].values.tolist()

    Prediction = CSV['Final Prediction'].values.tolist()

    # Truth_Prediction
    Positive_Correct = 0
    False_Positive = 0
    Negative_Correct = 0
    False_Negative = 0
    for i in range(len(Truth)):

        if Truth[i] == 0 and Prediction[i] == 0:
            Negative_Correct += 1

        if Truth[i] == 1 and Prediction[i] == 1:
            Positive_Correct += 1

        if Truth[i] == 1 and Prediction[i] == 0:
            False_Negative += 1

        if Truth[i] == 0 and Prediction[i] == 1:
            False_Positive += 1

    print("True Positive:", Positive_Correct, "True Negative:", Negative_Correct)
    print("False Positives:", False_Positive, "False Negatives:", False_Negative)
    Table_Data = [[Positive_Correct, False_Positive], [False_Negative, Negative_Correct]]
    Columns = ('Positive', 'Negative')
    Rows = ('Positive', 'Negative')

    Con = plt.table(cellText=Table_Data, rowLabels=Rows, colLabels=Columns, loc='center')
    Con.set_fontsize(24)
    Con.scale(4, 4)

    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.grid(False)
    plt.savefig(Results_Folder + "\\Confusion Matrix", bbox_inches='tight', pad_inches=0.05)


Confusion_Matrix(Results_Folder + "\\" + "Results.csv")

###################################################### Make ROC Curve ##################################################

def ROC_Curve(CSV_Filepath):
    print("Generating ROC curve")
    CSV = pd.DataFrame(pd.read_csv(CSV_Filepath, usecols=['Ground Truth (1:Immature 0:Mature)', 'Immature Chance']))
    Ground_Truth = CSV['Ground Truth (1:Immature 0:Mature)'].values.tolist()
    Prediction = CSV['Immature Chance'].values.tolist()
    fpr, tpr, Thresh = roc_curve(Ground_Truth[:], Prediction[:])
    AUC = auc(fpr, tpr)
    return fpr, tpr, AUC


fpr, tpr, AUC = ROC_Curve(Results_Folder + "\\" + "Results.csv")

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig(Results_Folder + "\\ROC curve.png", bbox_inches='tight', pad_inches=0.05)
