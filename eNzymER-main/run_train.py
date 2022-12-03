import argparse
import eNzymER_model as ener

if __name__ == '__main__':

    wdEmbed = "BioBERT" # or "BioBERT"
    operation = "test"  # or "train"

    em = ener.eNzymER(wdEmbed)
    if operation == "train":
        if wdEmbed == "BioBERT":
            em.train('TrainingSet.txt', "TrainingSetAnnot.txt", "BioModel")
        else:
            em.train('TrainingSet.txt', "TrainingSetAnnot.txt", "SciModel")
    else:
        if wdEmbed == "BioBERT":
            em.load('eNzymER_BioModel.json', './BioBertModels/epoch_9_BioModel_weights')
        else:
            em.load('eNzymER_SciModel.json', './SciBertModels/epoch_9_SciModel_weights')
        em.test('test_exclude_abbre.txt', "testAnnotated_exclude_abbre.txt")
    
    """
    # Using command-lines
    parser = argparse.ArgumentParser(prog='PROG')

    parser.add_argument('-t', '--text_path', type=str)
    parser.add_argument('-a', '--annot_path', type=str)
    parser.add_argument('-o', '--output_name', type=str)

    args = parser.parse_args()
    em.train(args.text_path, args.annot_path, args.output_name)
    # e.g. python run_app.py -t "TrainingSet.txt" -a "TrainingSetAnnot.tsv" -o "eNzymERModel"c
    """
    
