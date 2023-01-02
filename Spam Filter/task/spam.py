# write your code here
from DataProcessor import DataProcessor
from LanguageProcessor import LanguageProcessor
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

def main():
    file_name = 'spam.csv'
    data_processor = DataProcessor(file_name)
    data_processor.load_csv()
    data_processor.process_data()
    lang_processor = LanguageProcessor(data_processor.processed_file)
    lang_processor.split_train_test()
    bag, model = lang_processor.train_model()
    #from sklearn
    sk_prediction = lang_processor.predict_df('sklearn', model)
    stats = lang_processor.prediction_statistics(sk_prediction)


    # Homemade

    #prob = lang_processor.get_probabilities(bag, 1)
    #predictions = lang_processor.predict_df('homemade')
    #stats = lang_processor.prediction_statistics(predictions)



    # pd.options.display.max_columns = 50
    # pd.options.display.max_rows = 50
    # df = predictions.loc[[1245, 1708, 747, 3744, 3293], :]
    # df = df.append(predictions.sample(n=33, random_state=43, ignore_index=False), ignore_index=False)
    # df = df.append(predictions.loc[[113, 3053, 5455, 3666, 1780], :], ignore_index=False)
    # df = df.append(predictions.sample(n=2, random_state=43, ignore_index=False), ignore_index=False)
    # df = df.append(predictions.loc[[2549, 3430, 3633, 4665, 4033], :])
    print(stats)




if __name__ == "__main__":
    main()
