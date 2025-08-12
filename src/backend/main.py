from model.model import PlayerClassifier
from preprocessing.data_cleaner import *

def main():

    players = ['joshua-kimmich', 'thomas-mueller', 'alphonso-davies', 'michael-olise', 'leon-goretzka']

    rename_and_clean_images(players)
    crop_face(players)

    classifier = PlayerClassifier(data_dir="data/cropped-images")
    classifier.train(num_epochs=10)
    classifier.evaluate()

    
if __name__ == "__main__":
    main()