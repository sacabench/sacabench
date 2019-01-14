# *******************************************************************************
# Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de
#                    Florian Grieskamp <florian.grieskamp@tu-dortmund.de
#
# All rights reserved. Published under the BSD-3 license in the LICENSE file.
#******************************************************************************

import sys      # Dependency for accessing arguments.
import json     # Dependency for processing the json files.
from jinja2 import Environment, FileSystemLoader # Parsing template files

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

########################################
# PROCESSING ALGORITHM DATA
########################################

def extractAlgorithmDataFromDictionary(dict, algorithmID, repetitionID):    
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
    
    data["algorithmID"] = algorithmID
    data["repetitionID"] = repetitionID

    data["title"] = dict["title"]
    data["memFinal"] = dict["memFinal"]
    data["memOff"] = dict["memOff"]
    data["memPeak"] = dict["memPeak"]

    timeEnd = dict["timeEnd"]
    timeStart = dict["timeStart"]
    data["time"] = timeEnd - timeStart

    stats = dict["stats"]
    for statsDict in stats:
        if statsDict["key"] == "input_file":
            data["inputFile"] = statsDict["value"]
        if statsDict["key"] == "repetitions":
            data["repetitionCount"] = statsDict["value"]
        if statsDict["key"] == "prefix":
            data["prefix"] = statsDict["value"]

    # Access first item in list of subs to get needed values. 
    # Every item of sub list contains these values.
    exampleSub = dict["sub"][0]
    exampleStats = exampleSub["stats"]
    for statsDict in exampleStats:
        if statsDict["key"] == "extra_sentinels":
            data["extraSentinels"] = statsDict["value"]
        if statsDict["key"] == "sa_index_bit_size":
            data["saIndexBitSize"] = statsDict["value"]
        if statsDict["key"] == "text_size":
            data["textSize"] = statsDict["value"]

    numberOfPhases = 0
    allPhases = dict["sub"][0]["sub"]
    for phases in allPhases:
        # Count only sub-phases of algorithm without preperations.
        if len(phases["sub"]) > 0:
            algorithmPhases = phases["sub"]
            numberOfPhases = len(algorithmPhases)
    data["numberOfPhases"] = numberOfPhases

    return data
            
def buildAlgorithmResultString(data):
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
    content += "algo={}\t".format(data["title"])
    content += "algorithm_name={}\t".format(data["title"])
    content += "id={}\t".format(data["algorithmID"])
    content += "rep={}\t".format(data["repetitionCount"])
    content += "rep_id={}\t".format(data["repetitionID"])
    content += "phases={}\t".format(data["numberOfPhases"])
    content += "extra_sentinels={}\t".format(data["extraSentinels"])
    content += "input={}\t".format(data["inputFile"])
    content += "prefix={}\t".format(data["prefix"])
    content += "sa_index_bit_size={}\t".format(data["saIndexBitSize"])
    content += "text_size={}\t".format(data["textSize"])
    content += "memOff={}\t".format(data["memOff"])
    content += "memFinal={}\t".format(data["memFinal"])
    content += "memPeak={}\t".format(data["memPeak"])
    # TODO: Number of threads is currently not included in input JSON. 
    content += "threads=1\t" 
    content += "time={}\n".format(data["time"])

    return content	

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
    data["algorithmID"] = algorithmID
    data["repetitionID"] = repetitionID
    data["phaseID"] = phaseID

    data["title"] = dict["title"]
    data["memFinal"] = dict["memFinal"]
    data["memOff"] = dict["memOff"]
    data["memPeak"] = dict["memPeak"]

    timeEnd = dict["timeEnd"]
    timeStart = dict["timeStart"]
    data["time"] = timeEnd - timeStart

    return data	

def buildPhasesResultString(data):
    """
    Creates a line for the phases result text file.

    Parameters
    ----------
    dict : dictionary
        The dictionary which contains the data for a phase.

    Returns
    ----------
        A string which represents one line of the phases result text file.
    """

    content = "RESULT\t"
    content += "id={}\t".format(data["algorithmID"])
    content += "rep_id={}\t".format(data["repetitionID"])
    content += "phase_id={}\t".format(data["phaseID"])
    content += "phase_name={}\t".format(data["title"])
    content += "memOff={}\t".format(data["memOff"])
    content += "memFinal={}\t".format(data["memFinal"])
    content += "memPeak={}\t".format(data["memPeak"])
    content += "time={}\n".format(data["time"])

    return content

########################################
# PROCESS FILE
########################################

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

    currentrepetitionNumber = 0
    currentAlgorithmNumber = 0
    currentPhaseNumber = 0

    for algorithm in dict:
        # Set id of current repetition to 0 when processing an new algorithm.
        currentrepetitionNumber = 0

        # Increase id of each algorithm.
        currentAlgorithmNumber += 1

        for repetition in algorithm:
            # Increase id of current repetition when processing a new repetition of the current algorith.
            currentrepetitionNumber += 1

            # Set id of current phase to 0 when processing an new repititon.
            currentPhaseNumber = 0

            algorithmDataDict = extractAlgorithmDataFromDictionary(repetition, currentAlgorithmNumber, currentrepetitionNumber)
            algorithmFileContent += buildAlgorithmResultString(algorithmDataDict)

            # List of sub contains only one element.
            allPhases = repetition["sub"][0]["sub"]
            for algorithmPhase in allPhases:
                # An algorithm contains next to its normal phases also some preparation phases.
                # Process only normal phases and ignore the preparation phases.
                possiblePhases = algorithmPhase["sub"]
                if len(possiblePhases) > 0:
                    for phase in possiblePhases:

                        currentPhaseNumber += 1

                        phaseDataDict = extractPhaseDataFromDictionary(phase, currentAlgorithmNumber, currentrepetitionNumber, currentPhaseNumber)
                        phasesFileContent += buildPhasesResultString(phaseDataDict)

    algorithmFilePath = "{}/result_algorithm.txt".format(path)
    writeFile(algorithmFilePath, algorithmFileContent)

    phasesFilePath = "{}/result_phases.txt".format(path)
    writeFile(phasesFilePath, phasesFileContent)

########################################
# GENERATE TEX SOURCE CODE
########################################
class Config:
    def __init__(self, input_dict):
        algo_count = len(input_dict)
        self.bar_width = 160 / algo_count

        configuration_dict = input_dict[0][0]["stats"]
        for configuration_entry in configuration_dict:
            if configuration_entry["key"] == "repetitions":
                self.repetition_count = configuration_entry["value"]
            if configuration_entry["key"] == "prefix":
                self.prefix = configuration_entry["value"]

def generate_tex(input_dict):
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
    template = env.get_template('main.tex')

    config = Config(input_dict)

    output_file = open('main.tex', 'w')
    output_file.write(template.render(config=config))

########################################
# MAIN
########################################
    
def main(sourceFilePath, destinationFilePath):
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

    inputDataDict = readJSON(sourceFilePath)
    convertAndSaveData(inputDataDict, destinationFilePath)
    generate_tex(inputDataDict)

if __name__ == "__main__":
    
    # Get source and destination file path from given arguments.
    sourceFilePath = sys.argv[1]
    destinationFilePath = sys.argv[2]

    # start main function
    main(sourceFilePath, destinationFilePath)
