# Check out these notebooks for inspirations: https://bit.ly/3u5WOac, https://bit.ly/3CVqZF3
#For instance, below are some dependencies you might need if you are using Pytorch
# FUTURE: Consider implementing a learning rate scheduler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import boto3
# from sagemaker.s3 import S3Downloader
# from sagemaker.session import Session
import sagemaker
import os
# import s3fs
from pickle import dump

from sagemaker import get_execution_role
# session = sagemaker.Session()


os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

sagemaker_session = sagemaker.Session(boto3.session.Session(region_name='us-east-1'))
bucket = sagemaker_session.default_bucket()
s3 = boto3.client('s3')

# # pass in the sagemaker session as an argument
# sagemaker_execution_role = get_execution_role(sagemaker_session=sagemaker_session)

    
import argparse

#TODO: Import dependencies for Debugging andd Profiling
# There's no debugging for HPO: https://github.com/awslabs/sagemaker-debugger/issues/325
# from smdebug import modes
# import smdebug.pytorch as smd
# from smdebug.pytorch import get_hook
from smdebug.profiler.utils import str2bool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, test_loader, criterion, epochs=1): # we don't need epochs for test set since we're just evaluating the trained model's performance
    for e in range(epochs):
        print('Epoch ', e + 1, '/', epochs)
        
        model.eval() #setting the model in evaluation mode
        
        # Setting the SMDebug hook for the testing / validation phase. 
        
        # hook.set_mode(smd.modes.EVAL)
        correct = 0
        num_batches = 0
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
                num_batches += 1
                

            # Record the testing loss and accuracy
            total_loss = running_loss / num_batches
            total_acc = correct / len(test_loader.dataset)
            
            print(f'Test Correct: {correct}')
            print(f'Test batch number: {num_batches}')
            print(f"Test Accuracy: {100*total_acc}, Test set: Average loss: {total_loss} at epoch: {e+1}")

def train(model, train_loader, validation_loader, criterion, optimizer, epochs):

    pickle_path = "/opt/ml/output/"
    class_pickle_filename = "dog_breeds_labels.pkl"
    saved_data = os.path.join(pickle_path, class_pickle_filename)
    class_names = [item[4:].replace("_", " ") for item in train_loader.dataset.classes]
    
    print("Printing and saving classes: \n")
    print(class_names[:5])
    print(f"Saving to this directory: {pickle_path}")
    
    # save as a pickle file
    with open(saved_data, 'wb') as f:
        dump(class_names, f)

    # Save it to s3 | see boto3 docs here: https://bit.ly/3IQXCFh
    with open(saved_data, 'rb') as data:
        s3.upload_fileobj(data, bucket, f'dogImages/classes/{class_pickle_filename}')
    
    
#     # Copying the file to an s3 bucket
#     ! aws s3 cp {pickle_path}{class_pickle_filename} s3://{bucket}/dogImages/classes/
    
    for e in range(epochs):
        print('Epoch ', e + 1, '/', epochs)
        
        model.train()
        
        # Setting the SMDebug hook for the training phase. #

        # hook.set_mode(smd.modes.TRAIN)
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
        total_loss = running_loss / num_batches
        
        print(f'Training Correct: {correct}')
        print(f'Training batch number: {num_batches}')
        print(f"Training Accuracy: {100*total_acc}, Training set: Average loss: {total_loss} at epoch: {e+1}")
        
        # Setting up our validation
        correct = 0
        num_batches = 0
        running_loss = 0.0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(validation_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Record the correct predictions for training data 
                running_loss += loss.item() # Accumulate the loss
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                num_batches += 1

            # Record the training loss and accuracy
            total_acc = correct / len(validation_loader.dataset)
            total_loss = running_loss / num_batches
            
            print(f'Validation Correct: {correct}')
            print(f'Validation Dataset length at batch: {len(validation_loader.dataset)}')
            print(f"Validation Accuracy: {100*total_acc}, Validation set: Average loss: {total_loss} at epoch: {e+1}")

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
    
# def net():
#     '''
#     TODO: Complete this function that initializes your model
#           Remember to use a pretrained model
#     '''
#     pass

    

def main(args):
  
    '''
    TODO: Initialize a model by calling the net function
    '''
    # model=net()
    params_to_update, model=net(model_name="squeezenet")
    print(model) #let's understand our model architecture

    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)
    
    # Registering the SMDebug hook to save output tensors. #
    
    # hook = smd.Hook.create_from_json_file()
    # hook.register_hook(model)
    # hook.register_loss(criterion)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
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

    
#     train(model, train_loader, criterion, optimizer)
    
    
    '''
    TODO: Test the model to see its accuracy
    '''
#     test(model, test_loader, criterion)
    
    
    epochs = args.epochs
    train(model, train_loader, validation_loader, criterion, optimizer, epochs)
    test(model, test_loader, criterion, epochs)
        

    print("Printing and saving model dict: \n")
    print(model.state_dict())
    print(f"Saving to this model directory: {os.environ['SM_MODEL_DIR']}")
    
    # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'dogs_classification_hpo.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':
 
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gpu", type=str2bool, default=True)
    parser.add_argument("--model_name", type=str, default="squeezenet")
    parser.add_argument("--num_classes", type=int, default=True)
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
        default=5, # change as needed
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
