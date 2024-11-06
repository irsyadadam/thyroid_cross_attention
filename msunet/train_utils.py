import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

class SegmentationDataset(Dataset):
    def __init__(self, map, targetSize = None):
        self.map = pd.read_csv(map, index_col=0)
        self.imageFilenames = self.map[self.map['new_path'].str.endswith('.npy') & ~self.map['new_path'].str.endswith('_m.npy')]['new_path'].values
        self.maskFilenames = [f"{i.split('.')[0]}_m.npy" for i in self.imageFilenames]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.targetSize = targetSize
    
    def __len__(self):
        return len(self.imageFilenames)

    def __getitem__(self, idx):
        image = processImage(self.imageFilenames[idx],self.targetSize)
        mask = processImage(self.maskFilenames[idx],self.targetSize)
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image, mask

class DiceLoss(nn.Module):
    def __init__(self, eps=1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + self.eps) / (m1.sum(1) + m2.sum(1) + self.eps)
        return 1 - score.sum() / num

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, outputs, targets, smooth=1):
        outputs = torch.sigmoid(outputs)       
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (outputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(outputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(outputs, targets, reduction='mean')
        if self.weight is not None:
            Dice_BCE = (1-self.weight)*BCE + (self.weight)*dice_loss
        else:
            Dice_BCE = (BCE + dice_loss)/2.0
        
        return Dice_BCE


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, focal_weight=0.5, pos_weight=torch.tensor([10]), gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, outputs, targets, smooth=1):

        # Binary cross entropy with logits
        BCE = F.binary_cross_entropy_with_logits(outputs, targets, pos_weight=self.pos_weight)

        outputs = torch.sigmoid(outputs)
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Dice Loss
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / (outputs.sum() + targets.sum() + smooth)

        # Binary Cross-Entropy Loss
        #BCE = F.binary_cross_entropy(outputs, targets, reduction='mean')

        # Focal Loss
        BCE_exp = torch.exp(-BCE)
        focal_loss = (1 - BCE_exp)**self.gamma * BCE

        # Combined Loss
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * BCE + self.focal_weight * focal_loss

        return combined_loss




# class TLoss(nn.Module):
#     def __init__(self, targetSize: int, device, nu: float = 1.0, epsilon: float = 1e-8, reduction: str = "mean"):
#         """
#         Implementation of the TLoss.

#         Args:
#             targetSize: One dimension of image size (int).
#             device: GPU device.
#             nu (float): Value of nu.
#             epsilon (float): Value of epsilon.
#             reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
#                              'none': no reduction will be applied,
#                              'mean': the sum of the output will be divided by the number of elements in the output,
#                              'sum': the output will be summed.
#         """
#         super().__init__()
#         self.D = torch.tensor((targetSize * targetSize), dtype=torch.float, device=device)
 
#         self.lambdas = torch.ones((targetSize, targetSize), dtype=torch.float, device=device)
#         self.nu = nn.Parameter(torch.tensor(nu, dtype=torch.float, device=device))
#         self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=device)
#         self.reduction = reduction

#     def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             input_tensor (torch.Tensor): Model's prediction, size (B x W x H).
#             target_tensor (torch.Tensor): Ground truth, size (B x W x H).

#         Returns:
#             torch.Tensor: Total loss value.
#         """

#         delta_i = input_tensor - target_tensor
#         sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
#         first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
#         second_term = torch.lgamma(sum_nu_epsilon / 2)
#         third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
#         fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
#         fifth_term = (self.D / 2) * (self.nu + self.epsilon)

#         delta_squared = torch.pow(delta_i, 2)
#         lambdas_exp = torch.exp(self.lambdas + self.epsilon)
#         numerator = delta_squared * lambdas_exp
#         numerator = torch.sum(numerator, dim=(1, 2))

#         fraction = numerator / sum_nu_epsilon
#         sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

#         total_losses = (first_term + second_term + third_term + fourth_term + fifth_term + sixth_term)

#         if self.reduction == "mean": return total_losses.mean()
#         elif self.reduction == "sum": return total_losses.sum()
#         elif self.reduction == "none": return total_losses
#         else: raise ValueError(f"The reduction method '{self.reduction}' is not implemented.")



def processImage(npyFilePath: str, targetSize: tuple=None):
    """
        Processes a grayscale image stored as a NumPy .npy file, performs scaling, type conversion, 
        resizing, and cropping/padding to return a square PIL image of specified target size.

        Parameters:
        - npyFilePath (str): The file path to the .npy file containing the image data.
        - targetSize (tuple): The desired size of the final image. If None, the final image will be 512x512.

        Returns:
        - PIL.Image: The processed PIL Image.
    """ 
    npImg = np.load(npyFilePath)
    if(npImg.dtype != np.uint8):
        if(np.max(npImg) - np.min(npImg) != 0): npImg = (255 * (npImg - np.min(npImg)) / (np.max(npImg) - np.min(npImg))).astype(np.uint8)
        else: npImg = (255*npImg).astype(np.uint8)
    pilImg = Image.fromarray(npImg, 'L')
    if targetSize is None: targetSize = (512,512)
    scale = targetSize[0]/max(pilImg.width, pilImg.height)
    if pilImg.width > pilImg.height: newSize = (targetSize[0], int(pilImg.height * scale))
    elif pilImg.width < pilImg.height: newSize = (int(pilImg.width * scale), targetSize[1])
    else: newSize = targetSize
    pilImg = pilImg.resize(newSize, Image.LANCZOS)
    finalImg = Image.new('L', targetSize)
    pastePosition = ((targetSize[0]-newSize[0])//2, (targetSize[1]-newSize[1])//2)
    finalImg.paste(pilImg, pastePosition)
    return finalImg

def get_sensitivity(SR,GT,threshold=0.5): # TPR, recall
    SR = SR > threshold
    GT = GT == 1
    TP = torch.sum(SR & GT)
    FN = torch.sum(~SR & GT)
    SE = float(TP/(TP + FN + 1e-8))
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == 1
    TN = torch.sum(~SR & ~GT)
    FP = torch.sum(SR & ~GT)
    SP = float(TN/(TN + FP + 1e-8))
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == 1
    TP = torch.sum(SR & GT)
    FP = torch.sum(SR & ~GT)
    PC = float(TP/(TP + FP + 1e-8))
    return PC

def get_FPR(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == 1
    TN = torch.sum(~SR & ~GT)
    FP = torch.sum(SR & ~GT)
    FPR = float(FP/(TN + FP + 1e-8))
    return FPR

def get_JS(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == 1
    Inter = torch.sum(SR & GT)
    Union = torch.sum(SR | GT)
    JS = float(Inter)/(float(Union) + 1e-8)
    return JS

def get_DC(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == 1
    Inter = torch.sum(SR & GT)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-8)
    return DC

def get_f1(SR, GT, threshold=0.5):
    precision = get_precision(SR,GT,threshold)
    recall = get_sensitivity(SR,GT,threshold)
    f1 = (2*precision*recall) / (precision + recall + 1e-8)
    return f1

def training(model, dataloader, optimizer, criterion, scheduler, test_loader, device, FOLD_NUM, epochs=50, model_file_ext=""):
    trainLoss = []
    trainDICE = []
    testLoss = []
    testDICE = []
    if not os.path.exists("logs/FOLD_%s_log.txt" % FOLD_NUM):
        with open("logs/FOLD_%s_log.txt" % FOLD_NUM, "w") as f:
            f.write("Training logs\n")
    best_loss = np.inf
    flip = 1
    L1 = nn.L1Loss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        save = False

        # tqdm progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, data in progress_bar:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update tqdm progress bar
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})

        epoch_loss = running_loss / len(dataloader)
        if epoch_loss < best_loss:  # save the best model
            best_loss = epoch_loss
            if flip == 1:
                torch.save(model.state_dict(), f'{model_file_ext}_v1.pth')
                model_version = "model v1"
                flip = 2
            else:
                torch.save(model.state_dict(), f'{model_file_ext}_v2.pth')
                model_version = "model v2"
                flip = 1
            save = True

        scheduler.step()

        trDICE = test(model,dataloader, device)
        teDICE = test(model,test_loader,device)
        teLoss = test_loss(model, test_loader, criterion, device)

        now = datetime.now()
        formatted_date = now.strftime("%m-%d %H:%M:%S")
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.6f}, Test Loss: {teLoss:.6f}, TrainDICE: {trDICE:.6f}, TestDICE: {teDICE:.6f}")
        with open("logs/logs.txt", "a") as f:
            f.write(f"TIMESTAMP: {formatted_date}, Epoch {epoch+1}, Train Loss: {epoch_loss:.6f}, Test Loss: {teLoss:.6f}, TrainDICE: {trDICE:.6f}, TestDICE: {teDICE:.6f}\n")
            if save: f.write("{model_version} saved.\n")

        trainLoss.append(epoch_loss)
        trainDICE.append(trDICE)
        testLoss.append(teLoss)
        testDICE.append(teDICE)

    return trainLoss, testLoss, trainDICE, testDICE

def test_loss(model, dataloader, criterion, device):
    model.eval()
    runningLoss = 0.0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            runningLoss += loss.item()
    return (runningLoss/len(dataloader))


def test(model, dataloader, device, verbose=False):
    model.eval()

    total = len(dataloader)
    sensitivity = 0
    specificity = 0
    precision = 0
    tpr = 0 # sensitivity, recall
    fpr = 0
    jaccard = 0
    dice = 0
    f1 = 0
    thr = 0.5
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs).view(-1)
            labels = labels.view(-1)
            sensitivity += get_sensitivity(outputs,labels,threshold=thr)
            specificity += get_specificity(outputs,labels,threshold=thr)
            precision += get_precision(outputs,labels,threshold=thr)
            tpr += sensitivity
            fpr += get_FPR(outputs,labels,threshold=thr)
            jaccard += get_JS(outputs,labels,threshold=thr)
            dice += get_DC(outputs,labels,threshold=thr)
            f1 += get_f1(outputs, labels, threshold=thr) 

    if(verbose):
        print(f"Sensitivity  : {sensitivity / total:.6f} \t Jaccard Score: {jaccard / total:.6f}")
        print(f"Specficity   : {specificity / total:.6f} \t DICE Score   : {dice / total:.6f}")
        print(f"Precision    : {precision / total:.6f} \t F1 Score     : {f1 / total:.6f}")
        print(f"FPR          : {fpr / total:.6f}")
    # print(f"TPR = Sensitivity = Recall")
    return (dice/total)

def save_predictions(model, image_dir, device):
    model.eval()
    model.to(device)
    thr = 0.5
    save_dir = "predictions"
    image_filenames = os.listdir(image_dir)
    transform = transforms.Compose([transforms.ToTensor()])
    
    for i in range(len(image_filenames)):
        img_name = os.path.join(image_dir, image_filenames[i])
        image = processImage(img_name)
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        output = model(image)
        prediction = (torch.sigmoid(output)>thr).squeeze().cpu().numpy()
        pred_img = Image.fromarray((prediction * 255).astype(np.uint8))
        pred_img.save(os.path.join(save_dir, f"{image_filenames[i].split('.')[0]}_pred.jpg"))
    
    print("Finished saving all predictions!")


def visualize_predictions(model, dataloader, device, num_images_display=4):
    print("Visualize")
    model.eval()
    model.to(device)

    fig, axs = plt.subplots(4, num_images_display, figsize=(20, 10))

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.squeeze().cpu().numpy()
            outputs = model(images)
            preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
            predicts = (torch.sigmoid(outputs)>0.5).squeeze().cpu().numpy()
            images = images.squeeze().cpu().numpy()
            for j in range(num_images_display):
                axs[0, j].imshow(images[j], cmap='gray')
                axs[0, j].set_title(f'Input Image {j}')
                axs[0, j].axis('off')
            
                axs[1, j].imshow(masks[j], cmap='gray')
                axs[1, j].set_title(f'Ground Truth {j}')
                axs[1, j].axis('off')

                axs[2, j].imshow(preds[j], cmap='gray')
                axs[2, j].set_title(f'Prediction {j}')
                axs[2, j].axis('off')

                axs[3, j].imshow(predicts[j], cmap='gray')
                axs[3, j].imshow(masks[j], cmap='Greens', alpha=0.3)
                axs[3, j].set_title(f'Overlay {j}')
                axs[3, j].axis('off')
            break

    plt.tight_layout()
    plt.savefig('visualizations.png', dpi=500)
    plt.close()