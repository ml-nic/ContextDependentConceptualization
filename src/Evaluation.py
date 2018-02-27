from Conceptualizer import Conceptualizer
from LDA import LDA

MODEL_BASE_DIR = 'F:/Not_Uploaded/conceptualization_eval/models/'
MODEL_FILE_EXTENSION = '.gensim'
model_names = [
    'ldamodel_50',
    'ldamodel_topics100_trainiter20_en_noStopWords',
    'ldamodel_topics100_trainiter20_full_en',
    'ldamodel_topics100_trainiter20_train_en',
    'ldamodel_topics100_trainiter20_train_en_keep_all'
]

for model_name in model_names:
    print("\nTest", model_name, ':')
    lda = LDA()
    lda.load(MODEL_BASE_DIR + model_name + MODEL_FILE_EXTENSION)
    conceptualizer = Conceptualizer(lda)

    context_instances = [
        ('When was Barack Obama born?', 'Barack Obama', ['person']),
        ('When was Michelle Obama born?', 'Michelle Obama', ['person']),
        ('Who is the wife of Barack Obama?', 'Barack Obama', ['person']),
        ('where is the headquarter of apple?', 'apple', ['firm', 'company']),
        ('Apple reveals new iPad', 'apple', ['firm', 'company']),
        ('how many people are there in honolulu?', 'honolulu', ['city']),
        ('how many people are there in waldshut-tiengen?', 'waldshut-tiengen', ['city']),
        ('how many people live in honolulu?', 'honolulu', ['city']),
        ('how many people in waldshut-tiengen?', 'waldshut-tiengen', ['city']),
        ('Which NFL team represented the AFC at Super Bowl 50?', 'NFL', ['']),
        ('Which NFL team represented the AFC at Super Bowl 50?', 'Super Bowl', [''])
        #('I cook them in the oven', 'apply heat'),
        #('I cook them in the oven', 'cooker'),
        #('I cook them in the oven', 'food'),
        #('I cook them in the oven', 'heat source')
    ]

    for element in context_instances:
        estimated_concept = conceptualizer.conceptualize(element[0], element[1])
        #if estimated_concept in element[2]:
        print(estimated_concept in element[2], element, estimated_concept)
