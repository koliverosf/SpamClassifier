# write your code here
from DataProcessor import DataProcessor
from LanguageProcessor import LanguageProcessor
import pandas as pd

def main():
    file_name = 'spam.csv'
    data_processor = DataProcessor(file_name)
    data_processor.load_csv()
    data_processor.process_data()
    lang_processor = LanguageProcessor(data_processor.processed_file)
    lang_processor.split_train_test()
    bag, model = lang_processor.train_model()

    #prob = lang_processor.get_probabilities(bag, 1)
    pd.options.display.max_columns = 50
    pd.options.display.max_rows = 200
    print(prob.iloc[:200, :3])




if __name__ == "__main__":
    main()