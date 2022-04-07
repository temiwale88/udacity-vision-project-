# Check out these notebooks for inspirations: https://bit.ly/3u5WOac, https://bit.ly/3CVqZF3
#For instance, below are some dependencies you might need if you are using Pytorch
# FUTURE: Consider implementing a learning rate scheduler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets

try: 
    import boto3
except:
    !pip install boto3
    print("boto3 is not available in current container, skipping...")

try: 
    import smdebug.pytorch as smd
except:
    print("smd is not available in current container, skipping...")


import sagemaker
import os

from sagemaker import get_execution_role
# session = sagemaker.Session()

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

sagemaker_session = sagemaker.Session(boto3.session.Session(region_name='us-east-1'))
bucket = sagemaker_session.default_bucket()

# # pass in the sagemaker session as an argument
# sagemaker_execution_role = get_execution_role(sagemaker_session=sagemaker_session)

    
import argparse
import time

#TODO: Import dependencies for Debugging andd Profiling

from smdebug import modes
from smdebug.pytorch import get_hook
from smdebug.profiler.utils import str2bool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, test_loader, criterion, hook):
       
    # Setting the SMDebug hook for the testing / validation phase. 
    
    hook.set_mode(smd.modes.EVAL)
    model.eval() #setting the model in evaluation mode

    correct = 0
    running_loss = 0.0
            
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Record the correct predictions for training data
            running_loss += loss.item() # Accumulate the loss 
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum() 

    # Record the testing loss and accuracy
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = correct / len(test_loader.dataset)
    
    print(f'Test Correct: {correct}')
    print(f"Test Accuracy: {100*total_acc}, Test set: Average loss: {total_loss}")
    
def train(model, train_loader, validation_loader, criterion, optimizer, epochs, hook):
    
    epoch_times = []
    
    if hook:
        hook.register_loss(criterion)
    
    
    print("START TRAINING")
    if hook:
        hook.set_mode(modes.TRAIN)
    start = time.time()
    model.train()
    
    # Setting the SMDebug hook for the training phase. #
    
    correct = 0
    num_batches = 0
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() # Reset gradients / slopes of cost w.r.t. the parameters to zero so they don't accumulate in memory. 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() #computes gradients
        optimizer.step() #backprop
        
        # Record the correct predictions for training data 
        running_loss += loss.item() # Accumulate the loss
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        num_batches += 1
        

    # Record the training loss and accuracy
    total_acc = correct / len(train_loader.dataset)
    train_loss = running_loss / len(train_loader.dataset) # Is this calculation correct?
    
    print(f'Training Correct: {correct}')
    print(f"Training Accuracy: {100*total_acc}, Training set: Average loss: {train_loss}")
    
    epoch_time = time.time() - start
    epoch_times.append(epoch_time)
    
    # calculate training time after all epoch
    p50 = np.percentile(epoch_times, 50)
    return p50

# Per documentation let's try Squeezenet: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False #This is the key part for transfer learning as we freeze the convolutional layers before we add our layer. Assumes that we set 'feature_extracting' = True for grad to be set to 'False'. See PyTorch documentation on transfer learning
            
def net(model_name, num_classes=133, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    # input_size = 0

    if model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract) #we've set preceding layers' requires_grad=False so there won't be any gradient computed for these layers. Only the next layers we declare.
        # model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)) #This is where we replace the classifier / last layer with our own to prediction our number of classes ('num_classes')
        
        # Another approach inspired by ywng: https://bit.ly/3u5WOac
        model_ft.classifier = nn.Sequential(
                            nn.Dropout(p=0.5),
                            nn.Conv2d(512, num_classes, kernel_size=1, stride=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(13)
                        )
        model_ft.num_classes = num_classes
        # input_size = 224
        

    else:
        print("Invalid model name, exiting...")
        exit()
        
    #     From: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("A few params_to_update")
                print("\t",name)
                
                """ This should be the expected printout per PyTorch's documentation:
                        Params to learn:
                        classifier.1.weight
                        classifier.1.bias
                """
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("Updating all parameters")
                print("\t",name)
    return params_to_update, model_ft
    
def main(args):

#     bucket = "sagemaker-studio-v5l3nlsfwc"
     
    # Download the data from s3
    train_key_prefix = "dogImages/train"
    train_folder_name = './dogImages/train'
    
    if not os.path.exists(train_folder_name):
        os.makedirs(train_folder_name)
        
    sagemaker_session.download_data(train_folder_name, bucket, train_key_prefix)
    
    test_key_prefix = "dogImages/test"
    test_folder_name = './dogImages/test'
    
    if not os.path.exists(test_folder_name):
        os.makedirs(test_folder_name)
    
    validation_key_prefix = "dogImages/valid"
    validation_folder_name = './dogImages/valid'
    
    if not os.path.exists(validation_folder_name):
        os.makedirs(validation_folder_name)
        
    sagemaker_session.download_data(train_folder_name, bucket, train_key_prefix)
    sagemaker_session.download_data(test_folder_name, bucket, test_key_prefix)
    sagemaker_session.download_data(validation_folder_name, bucket, validation_key_prefix)
    
    # DataLoaders and Datasets
    # From PyTorch's documentation on transfer learning
    # Data augmentation and normalization for training
    # Just normalization for validation and testing


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    
    # See helper codes from skysign on how to load this data: https://bit.ly/3MNdKe0

    train_data = datasets.ImageFolder(train_folder_name, transform=data_transforms['train'])
    validation_data = datasets.ImageFolder(validation_folder_name, transform=data_transforms['test']) #using the same transformation as in test set
    test_data = datasets.ImageFolder(test_folder_name, transform=data_transforms['test'])
    

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.train_batch_size, 
                                               num_workers=2,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=args.test_batch_size, 
                                               num_workers=2,
                                               shuffle=False)

    validation_loader = torch.utils.data.DataLoader(validation_data,
                                               batch_size=args.test_batch_size, # same as test
                                               num_workers=2,
                                               shuffle=False)

    print("Initializing Datasets and Dataloaders...")
    
    
    epochs = args.epochs

    # create model
    
    params_to_update, model=net(model_name="squeezenet")
    print(model) #let's understand our model architecture
    
    model.to(device)
    
    print(model)

    # Start the training.
    # ======================================================#
    # Register the SMDebug hook to save output tensors. #
    # ======================================================#
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # ===========================================================#
        # Pass the SMDebug hook to the train and test functions. #
        # ===========================================================#
        train(model, train_loader, validation_loader, criterion, optimizer, epoch, hook)
        test(model, test_loader, criterion, hook)

        # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'dogs_classification_hpo.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)
    
    median_time = train(model, train_loader, validation_loader, criterion, optimizer, epochs, hook)
    print("Median training time per Epoch=%.1f sec" % median_time)


if __name__=='__main__':
  
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--model", type=str, default="squeezenet")
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1, # change as needed
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--momentum", type=float, default=1.0, help="momentum (default: 0.9)" 
    )
  
    args = parser.parse_args()
    
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    main(args)
