import json
import csv


def create_csv(input_path = './dataset/data.json',output_path = './dataset/data.csv'):
    """a function that loads json file, and 
    returns a csv file containing 'intent' and 'pattern' in each row

    Parameters
    ----------
    input_path : str, optional
        the path of the json file, by default './data.json'
    output_path : str, optional
        the required path of the output csv file, by default './data.csv'
    """
    # Load data from input json
    file = open(input_path, encoding = 'utf8')
    json_content = json.load(file)


    # Create output file
    csv_file = open(output_path,'w')

    # Write into the output csv file
    writer = csv.writer(csv_file)
    print('======== Started CSV creation =========')
    writer.writerow(['pattern','intent'])
    for i in json_content['intents']:
        for j in (i['patterns']):
            row = [j,i['tag']]
            writer.writerow(row)
    print('======= Completed CSV creation ========')

    # Close the files
    csv_file.close()
    file.close()