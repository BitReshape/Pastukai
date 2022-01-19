import os
import csv
import pandas as pd


# create CSV file of all images 
def create_testing_file_csv(datatset_path, testing_set_csv_path):
    '''
    Function to create csv file out of testing dataset
    :param datatset_path: Path of the testing dataset
    :param testing_set_csv_path: Path to the testing csv file
    :return:
    '''
    filenames = []
    
    # Get all image filenames
    for filename in os.listdir(datatset_path):
        filenames.append(filename[:-4])
        
    # save image filenames in csv
    csvfilenames = []
    with open(testing_set_csv_path + 'testing.csv','w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image'])
        for file in filenames:
            if file[0:4] == "ISIC":
                writer.writerow([file])
                csvfilenames.append(file)     
    print("CSV file is created successfully.")
    

def create_results_file_csv(results_path, csv_name, images, predictions, max_predictions):
    '''

    :param results_path: Path to the results directory
    :param csv_name: Name of the csv file
    :param images: array of image names
    :param predictions: Array of predicted class
    :param max_predictions: Array with probability of the predictions
    :return:
    '''
    # check length of both files if they are equal
    if len(images) != len(predictions) and len(predictions) != len(max_predictions):
        print("!!!!! Length is not the same of the image array and prediction array !!!!!")
        
    # save imagesnames and predictions into csv file
    with open(results_path + csv_name + '.csv','w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'])
        for i in range(len(images)):
            if max_predictions[i] < 0.25:
                writer.writerow([images[i],0,0,0,0,0,0,0,0,1])
            else:
                if predictions[i] == 'MEL':
                    writer.writerow([images[i],1,0,0,0,0,0,0,0,0])
                elif predictions[i] == 'NV':
                    writer.writerow([images[i],0,1,0,0,0,0,0,0,0])
                elif predictions[i] == 'BCC':
                    writer.writerow([images[i],0,0,1,0,0,0,0,0,0])
                elif predictions[i] == 'AK':
                    writer.writerow([images[i],0,0,0,1,0,0,0,0,0])            
                elif predictions[i] == 'BKL':
                    writer.writerow([images[i],0,0,0,0,1,0,0,0,0])
                elif predictions[i] == 'DF':
                    writer.writerow([images[i],0,0,0,0,0,1,0,0,0])
                elif predictions[i] == 'VASC':
                    writer.writerow([images[i],0,0,0,0,0,0,1,0,0])
                elif predictions[i] == 'SCC':
                    writer.writerow([images[i],0,0,0,0,0,0,0,1,0])
                elif predictions[i] == 'UNK':
                    writer.writerow([images[i],0,0,0,0,0,0,0,0,1])
                else:
                    print('Error! this class is unknown! Number:', i, 'Prediction:', predictions[i], 'images:', images[i])
            

    print("CSV file is created successfully.")
                 
    
    
def getImageTestingNames(testing_set_csv_path):
    '''
    Get all testing images name
    :param testing_set_csv_path:
    :return: Array of image names
    '''
    df_testing = pd.read_csv(testing_set_csv_path)
    image_names = []
    
    for i, image in df_testing.iterrows():
        image_names.append(image['image'])
    
    return image_names



def getMaxPredictions(predicted_testing_prob):
    '''
    Get from the svm output the maximum probability for each image
    :param predicted_testing_prob: Probabilities for each class of every image
    :return: Array of max probabilities
    '''
    max_prediction = []
    for item in predicted_testing_prob:
        max_prediction.append(max(item))
    return max_prediction