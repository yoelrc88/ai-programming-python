#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Yoel Ramos Castillo
# DATE CREATED: Jun 14, 2018
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below
def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_args = get_input_args()
    # print("Argument 1:", in_args.dir, "  Argument 2:",
    #       in_args.arch, "  Argument 3:", in_args.dogfile)

    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_args.dir)
    # print("Pet Labels number: ", len(answers_dic))
    # count = 0
    # for key in answers_dic:
    #     count += 1
    #     print("%d- key: %-30s label: %-30s" % (count, key, answers_dic[key]))

    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_args.dir, answers_dic, in_args.arch)
    
    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_args.dogfile)

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_args.arch, True, True)

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\nTotal Elapsed Runtime:", str(int((tot_time / 3600))) + ":" +
          str(int(((tot_time % 3600) / 60))) + ":" +
          str(int(((tot_time % 3600) % 60))))



# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: path to pet_images
    parser.add_argument('--dir', type=str, default='pet_images',
                        help='path to the folder pet_images')

    # Argument 2: the cnn_model
    parser.add_argument('--arch', type=str, default='resnet',
                        help='resnet, alexnet, or vgg')

    # Argument : The file that contains the list of valid dognames
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='The file that contains the list of valid dognames')

    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()

    return in_args


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these label as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    in_files = listdir(image_dir)
    petlabels_dic = dict()

    for idx in range(0, len(in_files)):
        if in_files[idx][0] == ".":
            continue

        pet_label = ""
        pet_label_words = in_files[idx].lower().split("_")
        
        for word in pet_label_words:
            if word.isalpha():
                pet_label += word + " "

        pet_label = pet_label.strip()

        if in_files[idx] not in petlabels_dic:
            petlabels_dic[in_files[idx]] = pet_label
        else:
            print("Warning: Duplicate files exist in directory: ", in_files[idx])
    return petlabels_dic


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict()
    
    for image_name in petlabel_dic:
        
        truth = petlabel_dic[image_name]

        image_path = images_dir + "/" + image_name
        model_label = classifier(image_path, model).lower()

        found = model_label.strip(" ").find(truth)

        if found >= 0:
            if ((found == 0 and len(truth) == len(model_label)) or
                (   ((found == 0) or (model_label[found - 1] == " ")) and
                    ((found + len(truth) == len(model_label)) or
                    (model_label[found + len(truth): found+len(truth)+1] in
                            (",", " "))
                    )
                )
                ):
                results_dic[image_name] = [truth, model_label,1]
            else:
                results_dic[image_name] = [truth, model_label,0]
        else:
            results_dic[image_name] = [truth, model_label, 0]
    return results_dic


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """           
    dognames_dic = dict()

    with open(dogsfile, "r") as infile:
        # Reads in dognames from first line in file
        line = infile.readline()

        # Processes each line in file until reaching EOF (end-of-file) by
        # processing line and adding dognames to dognames_dic with while loop
        while line != "":

            # TODO: 5.a REPLACE pass with CODE remove the newline character from
            #           the variable line
            #
            # Process line by striping newline from line
            line = line.rstrip('\n')

            # TODO: 5.b REPLACE pass with CODE TO check if dogname(line) doesn't
            #           exist within dognames_dic, then add the dogname(line)
            #           to dognames_dic as the 'key' with the 'value' of 1.
            #
            # adds dogname to dogsnames_dic if it doesn't already exist in dic
            if line not in dognames_dic:
                dognames_dic[line] = 1
            else:
                print("Warning: Duplicate files exist in directory: ", line)

            # Reads in next line in file to be processed with while loop
            # if this line isn't empty (EOF)
            line = infile.readline()
    
    # Add to whether pet labels & classifier labels are dogs by appending
    # two items to end of value(List) in results_dic.
    # List Index 3 = whether(1) or not(0) Pet Image Label is a dog AND
    # List Index 4 = whether(1) or not(0) Classifier Label is a dog
    # How - iterate through results_dic if labels are found in dognames_dic
    # then label "is a dog" index3/4=1 otherwise index3/4=0 "not a dog"
    for key in results_dic:

        # Pet Image Label IS of Dog (e.g. found in dognames_dic)
        if results_dic[key][0] in dognames_dic:

            # Classifier Label IS image of Dog (e.g. found in dognames_dic)
            # appends (1, 1) because both labels are dogs
            if results_dic[key][1].lower() in dognames_dic:
                results_dic[key].extend((1, 1))

            # TODO: 5.c REPLACE pass BELOW with CODE that adds the following to
            #           results_dic dictionary for the key indicated by the
            #           variable key - append (1,0) to the value. This indicates
            #           the pet label is-a-dog, classifier label is-NOT-a-dog.
            #
            # Classifier Label IS NOT image of dog (e.g. NOT in dognames_dic)
            # appends (1,0) because only pet label is a dog
            else:
                results_dic[key].extend((1, 0))

        # Pet Image Label IS NOT a Dog image (e.g. NOT found in dognames_dic)
        else:
            # TODO: 5.d REPLACE pass BELOW with CODE that adds the following to
            #           results_dic dictionary for the key indicated by the
            #           variable key - append (0,1) to the value. This indicates
            #           the pet label is-NOT-a-dog, classifier label is-a-dog.
            #
            # Classifier Label IS image of Dog (e.g. found in dognames_dic)
            # appends (0, 1)because only Classifier labe is a dog
            if results_dic[key][1].lower() in dognames_dic:
                results_dic[key].extend((0, 1))

            # TODO: 5.e REPLACE pass BELOW with CODE that adds the following to
            #           results_dic dictionary for the key indicated by the
            #           variable key - append (0,1) to the value. This indicates
            #           the pet label is-NOT-a-dog, classifier label is-NOT-a-dog.
            #
            # Classifier Label IS NOT image of Dog (e.g. NOT in dognames_dic)
            # appends (0, 0) because both labels aren't dogs
            else:
                results_dic[key].extend((0, 0))


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    results_stats  = dict()

    # Sets all counters to initial values of zero so that they can
    # be incremented while processing through the images in results_dic
    results_stats['n_dogs_img'] = 0
    results_stats['n_match'] = 0
    results_stats['n_correct_dogs'] = 0
    results_stats['n_correct_notdogs'] = 0
    results_stats['n_correct_breed'] = 0

    # process through the results dictionary
    for key in results_dic:

        # Labels Match Exactly
        if results_dic[key][2] == 1:
            results_stats['n_match'] += 1

        # TODO: 6.a REPLACE pass with CODE that counts how many pet images of
        #           dogs had their breed correctly classified. This happens
        #           when the pet image label indicates the image is-a-dog AND
        #           the pet image label and the classifier label match. You
        #           will need to write a conditional statement that determines
        #           when the dog breed is correctly classified and then
        #           increments 'n_correct_breed' by 1. Recall 'n_correct_breed'
        #           is a key in the results_stats dictionary with it's value
        #           representing the number of correctly classified dog breeds.
        #
        # Pet Image Label is a Dog AND Labels match- counts Correct Breed
        if results_dic[key][2] == 1 and results_dic[key][3] == 1:
            results_stats['n_correct_breed'] += 1


        # Pet Image Label is a Dog - counts number of dog images
        if results_dic[key][3] == 1:
            results_stats['n_dogs_img'] += 1

            # Classifier classifies image as Dog (& pet image is a dog)
            # counts number of correct dog classifications
            if results_dic[key][4] == 1:
                results_stats['n_correct_dogs'] += 1

        # TODO: 6.b REPLACE pass with CODE that counts how many pet images
        #           that are NOT dogs were correctly classified. This happens
        #           when the pet image label indicates the image is-NOT-a-dog
        #           AND the classifier label indicates the images is-NOT-a-dog.
        #           You will need to write a conditional statement that
        #           determines when the classifier label indicates the image
        #           is-NOT-a-dog and then increments 'n_correct_notdogs' by 1.
        #           Recall the 'else:' above 'pass' already indicates that the
        #           pet image label indicates the image is-NOT-a-dog and
        #          'n_correct_notdogs' is a key in the results_stats dictionary
        #           with it's value representing the number of correctly
        #           classified NOT-a-dog images.
        #
        # Pet Image Label is NOT a Dog
        else:
            # Classifier classifies image as NOT a Dog(& pet image isn't a dog)
            # counts number of correct NOT dog clasifications.
            if results_dic[key][4] == 0:
              results_stats['n_correct_notdogs'] += 1

    # Calculates run statistics (counts & percentages) below that are calculated
    # using counters from above.

    # calculates number of total images
    results_stats['n_images'] = len(results_dic)

    # calculates number of not-a-dog images using - images & dog images counts
    results_stats['n_notdogs_img'] = (results_stats['n_images'] -
                                      results_stats['n_dogs_img'])

    # TODO: 6.c REPLACE zero(0.0) with CODE that calculates the % of correctly
    #           matched images. Recall that this can be calculated by the
    #           number of correctly matched images ('n_match') divided by the
    #           number of images. This result will need to be multiplied by
    #           100 to provide the percentage.
    #
    # Calculates % correct for matches
    results_stats['pct_match'] = results_stats['n_match'] / \
        results_stats['n_images'] * 100

    # TODO: 6.d REPLACE zero(0.0) with CODE that calculates the % of correctly
    #           classified dog images. Recall that this can be calculated by
    #           the number of correctly classified dog images divided by the
    #           number of dog images. This result will need to be multiplied by
    #           100 to provide the percentage.
    #
    # Calculates % correct dogs
    results_stats['pct_correct_dogs'] = results_stats['n_correct_dogs'] / \
        results_stats['n_dogs_img'] * 100

    # TODO: 6.e REPLACE zero(0.0) with CODE that calculates the % of correctly
    #           classified breeds of dogs. Recall that this can be calculated
    #           by the number of correctly classified breeds of dog divided by
    #           the number of dog images. This result will need to be
    #           multiplied by 100 to provide the percentage.
    #
    # Calculates % correct breed of dog
    results_stats['pct_correct_breed'] = results_stats['n_correct_breed'] / \
        results_stats['n_dogs_img'] * 100

    # Calculates % correct not-a-dog images
    # Uses conditional statement for when no 'not a dog' images were submitted
    if results_stats['n_notdogs_img'] > 0:
        results_stats['pct_correct_notdogs'] = (results_stats['n_correct_notdogs'] /
                                                results_stats['n_notdogs_img'])*100.0
    else:
        results_stats['pct_correct_notdogs'] = 0.0

    # returns results_stast dictionary
    return results_stats


def print_results(results_dic, results_stats, model, print_incorrect_dogs, print_incorrect_breed):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    # TODO: 7. EDIT and ADD code BELOW to do the following that's stated
    #          in the comments below that start with "TODO: 7."
    #
    # Prints summary statistics over the run
    print("\n\n*** Results Summary for CNN Model Architecture", model.upper(),
          "***")
    print("%20s: %3d" % ('N Images', results_stats['n_images']))
    print("%20s: %3d" % ('N Dog Images', results_stats['n_dogs_img']))

    # TODO: 7.a REPLACE print("") with CODE that prints the text string
    #          'N Not-Dog Images' and then the number of NOT-dog images
    #          that's accessed by key 'n_notdogs_img' using dictionary
    #          results_stats
    #
    print("N Not-Dog Images: %d" % results_stats['n_notdogs_img'])

    # Prints summary statistics (percentages) on Model Run
    print(" ")
    for key in results_stats:
        # TODO: 7.b REPLACE pass with CODE that prints out all the percentages
        #           in the results_stats dictionary. Recall that all
        #           percentages in results_stats have 'keys' that start with
        #           the letter p. You will need to write a conditional
        #           statement that determines if the key starts with the letter
        #           'p' and then you want to use a print statement to print
        #           both the key and the value. Remember the value is accessed
        #           by results_stats[key]
        #
        if key[0] == 'p':
            print("%s: %f %%" % (key, results_stats[key]))

    # IF print_incorrect_dogs == True AND there were images incorrectly
    # classified as dogs or vice versa - print out these cases
    if (print_incorrect_dogs and
                ((results_stats['n_correct_dogs'] + results_stats['n_correct_notdogs'])
                 != results_stats['n_images'])
            ):
        print("\nINCORRECT Dog/NOT Dog Assignments:")

        # process through results dict, printing incorrectly classified dogs
        for key in results_dic:

            # TODO: 7.c REPLACE pass with CODE that prints out the pet label
            #           and the classifier label from results_dic dictionary
            #           ONLY when the classifier function (classifier label)
            #           misclassified dogs specifically:
            #             pet label is-a-dog and classifier label is-NOT-a-dog
            #               -OR-
            #             pet label is-NOT-a-dog and classifier label is-a-dog
            #           You will need to write a conditional statement that
            #          determines if the classifier function misclassified dogs
            #          See 'Adjusting Results Dictionary' section in
            #         '13. Classifying Labels as Dogs' for details on the
            #          format of the results_dic dictionary. Remember the value
            #          is accessed by results_dic[key] and the value is a list
            #          so results_dic[key][idx] - where idx represents the
            #          index value of the list and can have values 0-4.
            #
            # Pet Image Label is a Dog - Classified as NOT-A-DOG -OR-
            # Pet Image Label is NOT-a-Dog - Classified as a-DOG
            if results_dic[key][3] != results_dic[key][4]:
                if results_dic[key][3] == 1:
                    print(key + ": Pet Image Label is a Dog - Classified as NOT-A-DOG")
                else:
                    print(key + ": Pet Image Label is NOT-a-Dog - Classified as a-DOG")

    # IF print_incorrect_breed == True AND there were dogs whose breeds
    # were incorrectly classified - print out these cases
    if (print_incorrect_breed and
                (results_stats['n_correct_dogs'] !=
                 results_stats['n_correct_breed'])
            ):
        print("\nINCORRECT Dog Breed Assignment:")

        # process through results dict, printing incorrectly classified breeds
        for key in results_dic:

            # Pet Image Label is-a-Dog, classified as-a-dog but is WRONG breed
            if (sum(results_dic[key][3:]) == 2 and
                    results_dic[key][2] == 0):
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0],
                                                           results_dic[key][1]))


                
                
# Call to main function to run the program
if __name__ == "__main__":
    main()
