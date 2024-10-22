# Defining label indices based on the 'letter' split when accessing the EMNIST dataset
# Note: There are 27 different classes where the 0th class represents "Not a Letter" and each of
# the remaining 26 letter classes corresponds to both a letter's lowercase and uppercase version

letters_labels_map = {
    0: 'N/A', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e',
    6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
    11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
    16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
    21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'
}