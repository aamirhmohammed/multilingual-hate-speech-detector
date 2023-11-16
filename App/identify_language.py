import fasttext
from language_code_converter import get_full_language_names 

class Identify_Language:

    def __init__(self):
        # train model with Facebook language translation data
        pretrained_lang_model = "Machine_Learning_Data\lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=1) 
        for prediction in predictions[0]:
            # convert tupled, iso-code format into a readable string
            if prediction.startswith('__label__'):
                return get_full_language_names(prediction[9:])
        # if predictions.contains('__label__'):
        #     predictions = predictions[9:]
        # return predictions[0]

if __name__ == '__main__':
    LANGUAGE = Identify_Language()
    lang = LANGUAGE.predict_lang("buenos dias, como estas senora?")
    print(lang)
