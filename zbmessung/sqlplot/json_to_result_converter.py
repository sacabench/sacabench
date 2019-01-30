# *******************************************************************************
# Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de
#                    Florian Grieskamp <florian.grieskamp@tu-dortmund.de
#
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
#******************************************************************************

import sys      # Dependency for accessing arguments.
import os
import json     # Dependency for processing the json files.
from jinja2 import Environment, FileSystemLoader # Parsing template files
import platform, subprocess  # Getting information about CPU

########################################
# FILE HELPER
########################################

def readJSON(filepath):
    """
    Imports the content of the json file at the given filepath.

    Parameters
    ----------
    filepath : str
        Path to the json file which will be read.

    Returns
    ----------
        The content of the json file as a dictionary.
    """

    with open(filepath) as file:
        data = json.load(file)
        return data

def writeFile(filepath, content):
    """
    Writes the given content to the given filepath.

    Parameters
    ----------
    filepath : str
        Path where the content is writen to.
    content: str
        String, which is writen to the file.
    """

    textFile = open(filepath, "w")
    textFile.write(content)
    textFile.close()

class ResultKeys:
    algo = "algo"
    algorithm_name = "algorithm_name"
    id = "id"
    rep = "rep"
    rep_id = "rep_id"
    phases = "phases"
    phase_id = "phase_id"
    phase_name = "phase_name"
    extra_sentinels = "extra_sentinels"
    input = "input"
    prefix = "prefix"
    thread_count = "thread_count"
    sa_index_bit_size = "sa_index_bit_size"
    text_size = "text_size"
    memOff = "memOff"
    memFinal = "memFinal"
    memPeak = "memPeak"
    threads = "threads"
    time = "time"	

def buildResultString(data):
    """
    Creates a line for the algorithm result text file.

    Parameters
    ----------
    dict : dictionary
        The dictionary which contains the data for a repititon of an algorithm.

    Returns
    ----------
        A string which represents one line of the algorithm result text file.
    """

    content = "RESULT\t"
    for key, value in data.items():
        content += "{}={}\t".format(key, value)
    content += "\n"
    return content

########################################
# PROCESSING ALGORITHM DATA
########################################

def extractAlgorithmDataFromDictionary(dict, algorithmID, repititionCount, repetitionID):    
    """
    Processes the given dictionary and returns a new dictionary.
    This new dicitionary contains the extracted data needed to create a line in the algorithm result file.

    Parameters
    ----------
    dict : dictionary
        The dictionary which contains the data for an algorithm.
    algorithmID : int
        The id of the current algorithm.
    repetitionID : int
        The id of the current repetition.

    Returns
    ----------
        A dictionary containing the data for one line in the algorithm result file.
    """
    
    data = {}
    
    data[ResultKeys.id] = algorithmID
    data[ResultKeys.rep] = repititionCount
    data[ResultKeys.rep_id] = repetitionID

    stats = dict["stats"]
    for statsDict in stats:
        if statsDict["key"] == "algorithm_name":
            data[ResultKeys.algorithm_name] = statsDict["value"]
            data[ResultKeys.algo] = statsDict["value"]
        if statsDict["key"] == "input_file":
            fullInputPath = statsDict["value"]
            data[ResultKeys.input] = os.path.basename(fullInputPath)
        if statsDict["key"] == "prefix":
            data[ResultKeys.prefix] = statsDict["value"]
        if statsDict["key"] == "thread_count":
            data[ResultKeys.thread_count] = statsDict["value"]

    # Access first item in list of subs to get needed values. 
    # Every item of sub list contains these values.
    exampleSub = dict["sub"][0]
    exampleStats = exampleSub["stats"]
    for statsDict in exampleStats:
        if statsDict["key"] == "extra_sentinels":
            data[ResultKeys.extra_sentinels] = statsDict["value"]
        if statsDict["key"] == "sa_index_bit_size":
            data[ResultKeys.sa_index_bit_size] = statsDict["value"]
        if statsDict["key"] == "text_size":
            data[ResultKeys.text_size] = statsDict["value"]
    
    algorithmDataDict = dict["sub"][0]["sub"]
    for algorithmEntry in algorithmDataDict:
        if algorithmEntry["title"] == "Algorithm":
            # We found values for algorithm!
            timeEnd = algorithmEntry["timeEnd"]
            timeStart = algorithmEntry["timeStart"]
            data[ResultKeys.time] = timeEnd - timeStart
            data[ResultKeys.memFinal] = algorithmEntry["memFinal"]
            data[ResultKeys.memOff] = algorithmEntry["memOff"]
            data[ResultKeys.memPeak] = algorithmEntry["memPeak"]

    numberOfPhases = 0
    allPhases = dict["sub"][0]["sub"]
    for phases in allPhases:
        # Count only sub-phases of algorithm without preperations.
        if len(phases["sub"]) > 0:
            algorithmPhases = phases["sub"]
            numberOfPhases = len(algorithmPhases)
    data[ResultKeys.phases] = numberOfPhases

    return data

########################################
# PROCESSING PHASES DATA
########################################

def extractPhaseDataFromDictionary(dict, algorithmID, repetitionID, phaseID):
    """
    Processes the given dictionary and returns a new dictionary.
    This new dicitionary contains the extracted data needed to create a line in the phases result file.

    Parameters
    ----------
    dict : dictionary
        The dictionary which contains the data for an algorithm.
    algorithmID : int
        The id of the current algorithm.
    repetitionID : int
        The id of the current repetition.
    phaseID : int
        The id of the current phase.

    Returns
    ----------
        A dictionary containing the data for one line in the phases result file.
    """
    
    data = {}
    data[ResultKeys.id] = algorithmID
    data[ResultKeys.rep_id] = repetitionID
    data[ResultKeys.phase_id] = phaseID

    data[ResultKeys.phase_name] = dict["title"]
    data[ResultKeys.memFinal] = dict["memFinal"]
    data[ResultKeys.memOff] = dict["memOff"]
    data[ResultKeys.memPeak] = dict["memPeak"]

    timeEnd = dict["timeEnd"]
    timeStart = dict["timeStart"]
    data[ResultKeys.time] = timeEnd - timeStart

    return data	

########################################
# PROCESS FILE
########################################

def handleAlgorithm(dict, currentAlgorithmNumber, currentRepetitionNumber):

    algorithmFileContent = ""
    phasesFileContent = ""

    repetitionCount = len(dict)

    for repetition in dict:
        # Increase id of current repetition when processing a new repetition of the current algorith.
        currentRepetitionNumber += 1

        # Set id of current phase to 0 when processing an new repititon.
        currentPhaseNumber = 0

        algorithmDataDict = extractAlgorithmDataFromDictionary(repetition, currentAlgorithmNumber, repetitionCount, currentRepetitionNumber)
        algorithmFileContent += buildResultString(algorithmDataDict)

        # List of sub contains only one element.
        allPhases = repetition["sub"][0]["sub"]
        for algorithmPhase in allPhases:
            # An algorithm contains next to its normal phases also some preparation phases.
            # Process only normal phases and ignore the preparation phases.
            possiblePhases = algorithmPhase["sub"]
            if len(possiblePhases) > 0:
                for phase in possiblePhases:

                    currentPhaseNumber += 1

                    phaseDataDict = extractPhaseDataFromDictionary(phase, currentAlgorithmNumber, currentRepetitionNumber, currentPhaseNumber)
                    phasesFileContent += buildResultString(phaseDataDict)

    return (algorithmFileContent, phasesFileContent)
                        
def convertAndSaveData(dict, path):
    """
    Processes the given dictionary and saves the created files to the directory of the given path.

    Parameters
    ----------
    dict : dictionary
        The dictionary which contains the data created by the saca benchmark tool.
    path : str
        Path of the directory in which the result files will be created.
    """

    algorithmFileContent = ""
    phasesFileContent = ""

    currentRepetitionNumber = 0
    repetitionCount = 0
    currentAlgorithmNumber = 0
    currentPhaseNumber = 0

    # mode construct
    if type(dict[0]) == type({}):
        algorithmFileContent, phasesFileContent = handleAlgorithm(dict, currentAlgorithmNumber, currentRepetitionNumber)
    
    # mode batch
    if type(dict[0]) == type([]):
        for algorithm in dict:
            # Set id of current repetition to 0 when processing an new algorithm.
            currentRepetitionNumber = 0

            # Increase id of each algorithm.
            currentAlgorithmNumber += 1

            newAlgorithmFileContent, newPhasesFileContent = handleAlgorithm(algorithm, currentAlgorithmNumber, currentRepetitionNumber)
            algorithmFileContent += newAlgorithmFileContent
            phasesFileContent += newPhasesFileContent

    algorithmFilePath = "{}/result_algorithm.txt".format(path)
    writeFile(algorithmFilePath, algorithmFileContent)

    phasesFilePath = "{}/result_phases.txt".format(path)
    writeFile(phasesFilePath, phasesFileContent)

########################################
# GENERATE TEX SOURCE CODE
########################################
class Config:
    def __init__(self, config_dict, input_dict):
        algo_count = len(input_dict)
        self.bar_width = 160 / algo_count

        print(config_dict)
        
        if config_dict["repetitions"]:
            self.repetition_count = config_dict["repetitions"]
        if config_dict["prefix"]:
            self.prefix = config_dict["prefix"]
        if config_dict["model_name"]:
            self.cpu = config_dict["model_name"]
        if config_dict["input"]:
            self.input_file = config_dict["input"]
            self.escaped_input_file = config_dict["input"].replace("_", "\_")

def generate_tex(config_dict, input_dict):
    file_loader = FileSystemLoader('templates')
    env = Environment(
            block_start_string = '\BLOCK{',
            block_end_string = '}',
            variable_start_string = '\VAR{',
            variable_end_string = '}',
            comment_start_string = '\#{',
            comment_end_string = '}',
            #line_statement_prefix = '%%',
            #line_comment_prefix = '%#',
            trim_blocks = True,
            autoescape = False,
            loader = file_loader)
    template = env.get_template('batch.tex')

    for count, configuration in enumerate(config_dict):
        config = Config(configuration, input_dict)
        output_file = open('batch-{}.tex'.format(count), 'w')
        output_file.write(template.render(config=config))

########################################
# MAIN
########################################
    
def main(plotConfigPath, sourceFilePath, destinationFilePath):
    """
    Main function for processing the result json of the saca benchmark tool.
    It converts the json at the given source file path into two result text files.
    These result files can be processed by SQLPlotTools.
    Saves the algorithm result file as "result_algorithm.txt" and 
    the phases result file as "result_phases.txt" in the given destination directory.

    Parameters
    ----------
    sourceFilePath : str
        The filepath of the json file generated by the saca benchmark tool.
    destinationFilePath : str
        The directory to which the two result files will be saved to.
    """

    configDict = readJSON(plotConfigPath)
    inputDataDict = readJSON(sourceFilePath)
    convertAndSaveData(inputDataDict, destinationFilePath)
    generate_tex(configDict, inputDataDict)

if __name__ == "__main__":
    
    # Get source and destination file path from given arguments.
    plotConfigPath = sys.argv[1]
    sourceFilePath = sys.argv[2]
    destinationFilePath = sys.argv[3]

    # start main function
    main(plotConfigPath, sourceFilePath, destinationFilePath)
